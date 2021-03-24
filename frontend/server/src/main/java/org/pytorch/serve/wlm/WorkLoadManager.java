package org.pytorch.serve.wlm;

import io.netty.channel.EventLoopGroup;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.pytorch.serve.snapshot.SnapshotManager;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.OSUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkLoadManager {

    private ExecutorService threadPool;

    private ConcurrentHashMap<ModelVersionName, List<WorkerThread>> workers;
    private ConcurrentHashMap<ModelVersionName, WorkerManagerThread> workerManagers;

    private ConfigManager configManager;
    private EventLoopGroup backendGroup;
    private AtomicInteger port;
    private AtomicInteger gpuCounter;

    private static final Logger logger = LoggerFactory.getLogger(WorkLoadManager.class);

    public WorkLoadManager(ConfigManager configManager, EventLoopGroup backendGroup) {
        this.configManager = configManager;
        this.backendGroup = backendGroup;
        this.port = new AtomicInteger(configManager.getInitialWorkerPort());
        this.gpuCounter = new AtomicInteger(0);
        threadPool = Executors.newCachedThreadPool();
        workers = new ConcurrentHashMap<>();
        workerManagers = new ConcurrentHashMap<>();
    }

    public List<WorkerThread> getWorkers(ModelVersionName modelVersionName) {
        List<WorkerThread> list = workers.get(modelVersionName);
        if (list == null) {
            return Collections.emptyList();
        }
        return new ArrayList<>(list);
    }

    public Map<Integer, WorkerThread> getWorkers() {
        Map<Integer, WorkerThread> map = new HashMap<>();
        for (List<WorkerThread> workerThreads : workers.values()) {
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

    public CompletableFuture<Integer> modelChanged(
            Model model, boolean isStartup, boolean isCleanUp) {
        synchronized (model.getModelVersionName()) {
            boolean isSnapshotSaved = false;
            CompletableFuture<Integer> future = new CompletableFuture<>();
            int minWorker = model.getMinWorkers();
            int maxWorker = model.getMaxWorkers();
            WorkerManagerThread workerManagerThread;
            List<WorkerThread> threads;
            if (minWorker == 0) {

                workerManagerThread = workerManagers.remove(model.getModelVersionName());
                threads = workers.remove(model.getModelVersionName());

                if (workerManagerThread != null) {
                    for (WorkerThread thread : threads) {
                        workerManagerThread.scaleDown(thread.getLifeCycle().getPort());
                        try {
                            String cmd =
                                    String.format(
                                            OSUtils.getKillCmd(thread.getLifeCycle().getPid()));
                            Process workerkillprocess = Runtime.getRuntime().exec(cmd, null, null);
                            workerkillprocess.waitFor(
                                    configManager.getUnregisterModelTimeout(), TimeUnit.SECONDS);
                        } catch (InterruptedException | IOException e) {
                            logger.warn(
                                    "WorkerManagerThread interrupted during waitFor, possible async resource cleanup.");
                            future.complete(HttpURLConnection.HTTP_INTERNAL_ERROR);
                            return future;
                        }
                    }
                    workerManagerThread.shutdown();
                    workerManagerThread.getLifeCycle().exit();

                    Process workerManagerProcess = workerManagerThread.getLifeCycle().getProcess();
                    if (workerManagerProcess != null && workerManagerProcess.isAlive()) {
                        boolean workerManagerDestroyed = false;
                        try {
                            String cmd =
                                    String.format(OSUtils.getKillCmd(workerManagerProcess.pid()));
                            Process workerkillprocess = Runtime.getRuntime().exec(cmd, null, null);
                            workerManagerDestroyed =
                                    workerkillprocess.waitFor(
                                            configManager.getUnregisterModelTimeout(),
                                            TimeUnit.SECONDS);
                        } catch (InterruptedException | IOException e) {
                            logger.warn(
                                    "WorkerManagerThread interrupted during waitFor, possible async resource cleanup.");
                            future.complete(HttpURLConnection.HTTP_INTERNAL_ERROR);
                            return future;
                        }
                        if (!workerManagerDestroyed) {
                            logger.warn(
                                    "WorkerManagerThread timed out while cleaning, please resend request.");
                            future.complete(HttpURLConnection.HTTP_CLIENT_TIMEOUT);
                            return future;
                        }
                    }
                }

                if (threads == null) {
                    future.complete(HttpURLConnection.HTTP_OK);
                    if (!isStartup && !isCleanUp && !model.isWorkflowModel()) {
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
                if (currentWorkers == 0) {
                    workerManagerThread =
                            workerManagers.computeIfAbsent(
                                    model.getModelVersionName(),
                                    k -> addWorkerManagerThread(model, future));
                }
                addThreads(threads, model, minWorker - currentWorkers, future);
            } else {

                workerManagerThread = workerManagers.get(model.getModelVersionName());

                if (workerManagerThread != null) {
                    for (int i = currentWorkers - 1; i >= maxWorker; --i) {
                        WorkerThread thread = threads.remove(i);
                        workerManagerThread.scaleDown(thread.getLifeCycle().getPort());
                        thread.shutdown();
                    }
                }
                if (!isStartup && !isCleanUp && !model.isWorkflowModel()) {
                    SnapshotManager.getInstance().saveSnapshot();
                    isSnapshotSaved = true;
                }
                future.complete(HttpURLConnection.HTTP_OK);
            }
            if (!isStartup && !isSnapshotSaved && !isCleanUp && !model.isWorkflowModel()) {
                SnapshotManager.getInstance().saveSnapshot();
            }
            return future;
        }
    }

    private WorkerManagerThread addWorkerManagerThread(
            Model model, CompletableFuture<Integer> future) {

        WorkerManagerStateListener listener = new WorkerManagerStateListener(future, 1);
        BatchAggregator aggregator = new BatchAggregator(model);
	int maxGpu = configManager.getNumberOfGpu();
        int gpuId = -1;
        if (maxGpu > 0) {
            gpuId = gpuCounter.accumulateAndGet(maxGpu, (prev, maxGpuId) -> ++prev % maxGpuId);
        }

        WorkerManagerThread thread =
                new WorkerManagerThread(
                        configManager,
                        backendGroup,
                        configManager.isDebug() ? port.get() : port.getAndIncrement(),
			gpuId,
                        model,
                        aggregator,
                        listener);
        threadPool.submit(thread);
        return thread;
    }

    private void addThreads(
            List<WorkerThread> threads, Model model, int count, CompletableFuture<Integer> future) {
        WorkerStateListener listener = new WorkerStateListener(future, count);
        for (int i = 0; i < count; ++i) {

            int portNum = port.getAndIncrement();
            workerManagers.get(model.getModelVersionName()).scaleUp(portNum);

            BatchAggregator aggregator = new BatchAggregator(model);
            WorkerThread thread =
                    new WorkerThread(
                            configManager,
                            backendGroup,
                            portNum,
			    workerManagers.get(model.getModelVersionName()).getGpuId(),
                            model,
                            aggregator,
                            listener);
            threads.add(thread);
            threadPool.submit(thread);
        }
    }

    public void scheduleAsync(Runnable r) {
        threadPool.execute(r);
    }
}
