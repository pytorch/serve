package org.pytorch.serve.wlm;

import io.netty.channel.EventLoopGroup;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import org.pytorch.serve.archive.ModelVersionNotFoundException;
import org.pytorch.serve.snapshot.SnapshotManager;
import org.pytorch.serve.util.ConfigManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkLoadManager {

    private ExecutorService threadPool;

    private ConcurrentHashMap<ModelVersionName, List<WorkerThread>> workers;

    private ConfigManager configManager;
    private EventLoopGroup backendGroup;
    private AtomicInteger port;
    private AtomicInteger gpuCounter;
    private AtomicInteger threadNumber;

    private static final Logger logger = LoggerFactory.getLogger(WorkLoadManager.class);

    public WorkLoadManager(ConfigManager configManager, EventLoopGroup backendGroup) {
        this.configManager = configManager;
        this.backendGroup = backendGroup;
        this.port = new AtomicInteger(9000);
        this.gpuCounter = new AtomicInteger(0);
        threadPool = Executors.newCachedThreadPool();
        workers = new ConcurrentHashMap<>();
        threadNumber = new AtomicInteger(0);
    }

    public List<WorkerThread> getWorkers(ModelVersionName modelVersionName) {
        List<WorkerThread> list = workers.get(modelVersionName);
        if (list == null) {
            return Collections.emptyList();
        }
        return new ArrayList<>(list);
    }

    public Map<Integer, WorkerThread> getWorkers() throws ModelVersionNotFoundException {
        Map<Integer, WorkerThread> map = new HashMap<>();
        for (Map.Entry<ModelVersionName, List<WorkerThread>> entry : workers.entrySet()) {
            // Add server thread
            String modelName = entry.getKey().getModelName();
            String modelVersion = entry.getKey().getVersion();
            List<WorkerThread> workerThreads = entry.getValue();
            WorkerThread serverThread =
                    ModelManager.getInstance().getModel(modelName, modelVersion).getServerThread();
            map.put(serverThread.getPid(), serverThread);

            for (WorkerThread worker : workerThreads) {
                map.put(worker.getPid(), worker);
            }
        }
        return map;
    }

    public boolean hasNoWorker(ModelVersionName modelVersionName) {
        List<WorkerThread> worker = workers.get(modelVersionName);
        if (worker == null) {
            return true;
        }
        return worker.isEmpty();
    }

    public int getNumRunningWorkers(ModelVersionName modelVersionName) {
        int numWorking = 0;
        List<WorkerThread> threads = workers.getOrDefault(modelVersionName, null);

        if (threads != null) {
            for (WorkerThread thread : threads) {
                if ((thread.getState() != WorkerState.WORKER_STOPPED)
                        && (thread.getState() != WorkerState.WORKER_ERROR)
                        && (thread.getState() != WorkerState.WORKER_SCALED_DOWN)) {
                    numWorking += 1;
                }
            }
        }

        return numWorking;
    }

    public CompletableFuture<HttpResponseStatus> modelChanged(Model model, boolean isStartup) {
        return modelChanged(model, isStartup, false);
    }

    public CompletableFuture<HttpResponseStatus> modelChanged(
            Model model, boolean isStartup, boolean isShutdown) {
        synchronized (model.getModelVersionName()) {
            boolean isSnapshotSaved = false;
            CompletableFuture<HttpResponseStatus> future = new CompletableFuture<>();
            int minWorker = model.getMinWorkers();
            int maxWorker = model.getMaxWorkers();
            List<WorkerThread> threads;
            if (minWorker == 0) {
                threads = workers.remove(model.getModelVersionName());
                if (threads == null) {
                    HttpResponseStatus stopThreadStatus = stopServerThread(model);
                    if (stopThreadStatus != HttpResponseStatus.OK) {
                        future.complete(stopThreadStatus);
                        return future;
                    }
                    future.complete(HttpResponseStatus.OK);
                    if (!isStartup) {
                        SnapshotManager.getInstance().saveSnapshot();
                    }
                    return future;
                }
            } else {
                threads =
                        workers.computeIfAbsent(
                                model.getModelVersionName(), k -> new ArrayList<>());
            }

            int currentWorkers = threads.size();
            if (currentWorkers < minWorker) {
                addThreads(threads, model, minWorker - currentWorkers, future);
            } else {
                for (int i = currentWorkers - 1; i >= maxWorker; --i) {
                    WorkerThread thread = threads.remove(i);
                    thread.shutdown();
                }
                if (maxWorker == 0) {
                    HttpResponseStatus stopThreadStatus = stopServerThread(model);
                    if (stopThreadStatus != HttpResponseStatus.OK) {
                        future.complete(stopThreadStatus);
                        return future;
                    }
                }
                if (!isStartup && !isShutdown) {
                    SnapshotManager.getInstance().saveSnapshot();
                    isSnapshotSaved = true;
                }
                future.complete(HttpResponseStatus.OK);
            }
            if (!isStartup && !isSnapshotSaved && !isShutdown) {
                SnapshotManager.getInstance().saveSnapshot();
            }
            return future;
        }
    }

    public void addServerThread(Model model, CompletableFuture<HttpResponseStatus> future)
            throws InterruptedException, ExecutionException, TimeoutException {
        WorkerStateListener listener = new WorkerStateListener(future, 1);
        BatchAggregator aggregator = new BatchAggregator(model);
        synchronized (model.getModelVersionName()) {
            model.setPort(port.getAndIncrement());
            WorkerThread thread =
                    new WorkerThread(
                            configManager,
                            backendGroup,
                            model.getPort(),
                            -1,
                            model,
                            aggregator,
                            listener,
                            threadNumber.getAndIncrement(),
                            true);
            model.setServerThread(thread);
            threadPool.submit(thread);
            future.get(1, TimeUnit.MINUTES);
        }
    }

    private HttpResponseStatus stopServerThread(Model model) {
        model.getServerThread().shutdown();
        WorkerLifeCycle lifecycle = model.getServerThread().getLifeCycle();

        Process workerProcess = lifecycle.getProcess();

        // Need to check worker process here since thread.shutdown() -> lifecycle.exit()
        // -> This may nullify process object per destroyForcibly doc.
        if (workerProcess != null && workerProcess.isAlive()) {
            boolean workerDestroyed = false;
            workerProcess.destroyForcibly();
            try {
                String cmd = String.format(getKillCmd(), workerProcess.pid());
                Process workerKillProcess = Runtime.getRuntime().exec(cmd, null, null);
                workerDestroyed =
                        workerKillProcess.waitFor(
                                configManager.getUnregisterModelTimeout(),
                                TimeUnit.SECONDS);
            } catch (InterruptedException | IOException e) {
                logger.warn(
                        "WorkerThread interrupted during waitFor, possible async resource cleanup.");
                future.complete(HttpResponseStatus.INTERNAL_SERVER_ERROR);
                return future;
            }
            if (!workerDestroyed) {
                logger.warn("WorkerThread timed out while cleaning, please resend request.");
                return HttpResponseStatus.REQUEST_TIMEOUT;
            }
        }
        return HttpResponseStatus.OK;
    }

    private String getKillCmd() {
        String operatingSystem = System.getProperty("os.name").toLowerCase();
        String killCMD;
        if (operatingSystem.indexOf("win") >= 0) {
            killCMD = "taskkill /f /PID %s";
        } else {
            killCMD = "kill -9 %s";
        }
        return killCMD;
    }

    private void addThreads(
            List<WorkerThread> threads,
            Model model,
            int count,
            CompletableFuture<HttpResponseStatus> future) {
        WorkerStateListener listener = new WorkerStateListener(future, count);
        int maxGpu = configManager.getNumberOfGpu();
        for (int i = 0; i < count; ++i) {
            int gpuId = -1;

            if (maxGpu > 0) {
                gpuId = gpuCounter.accumulateAndGet(maxGpu, (prev, maxGpuId) -> ++prev % maxGpuId);
            }

            BatchAggregator aggregator = new BatchAggregator(model);
            WorkerThread thread =
                    new WorkerThread(
                            configManager,
                            backendGroup,
                            model.getPort(),
                            gpuId,
                            model,
                            aggregator,
                            listener,
                            threadNumber.getAndIncrement(),
                            false);
            threads.add(thread);
            threadPool.submit(thread);
        }
    }

    public void scheduleAsync(Runnable r) {
        threadPool.execute(r);
    }
}
