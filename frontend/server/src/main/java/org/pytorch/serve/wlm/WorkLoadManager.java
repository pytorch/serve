package org.pytorch.serve.wlm;

import io.netty.channel.EventLoopGroup;
import io.netty.handler.codec.http.HttpResponseStatus;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkLoadManager {

    private ExecutorService threadPool;

    private ConcurrentHashMap<ModelVersionName, List<WorkerThread>> workers;

    private ConfigManager configManager;
    private EventLoopGroup backendGroup;
    private AtomicInteger port;
    private AtomicInteger gpuCounter;

    private static final Logger logger = LoggerFactory.getLogger(WorkLoadManager.class);

    public WorkLoadManager(ConfigManager configManager, EventLoopGroup backendGroup) {
        this.configManager = configManager;
        this.backendGroup = backendGroup;
        this.port = new AtomicInteger(9000);
        this.gpuCounter = new AtomicInteger(0);
        threadPool = Executors.newCachedThreadPool();
        workers = new ConcurrentHashMap<>();
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

    public CompletableFuture<HttpResponseStatus> modelChanged(Model model, boolean isStartup) {
        synchronized (model.getModelVersionName()) {
            boolean isSnapshotSaved = false;
            CompletableFuture<HttpResponseStatus> future = new CompletableFuture<>();
            int minWorker = model.getMinWorkers();
            int maxWorker = model.getMaxWorkers();
            List<WorkerThread> threads;
            if (minWorker == 0) {
                threads = workers.remove(model.getModelVersionName());
                if (threads == null) {
                    future.complete(HttpResponseStatus.OK);
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
                    WorkerLifeCycle lifecycle = thread.getLifeCycle();
                    thread.shutdown();

                    Process workerProcess = lifecycle.getProcess();

                    // Need to check worker process here since thread.shutdown() -> lifecycle.exit()
                    // -> This may nullify process object per destroyForcibly doc.
                    if (workerProcess != null && workerProcess.isAlive()) {
                        boolean workerDestroyed = false;
                        workerProcess.destroyForcibly();
                        try {
                            workerDestroyed =
                                    workerProcess.waitFor(
                                            configManager.getUnregisterModelTimeout(),
                                            TimeUnit.SECONDS);
                        } catch (InterruptedException e) {
                            logger.warn(
                                    "WorkerThread interrupted during waitFor, possible async resource cleanup.");
                            future.complete(HttpResponseStatus.INTERNAL_SERVER_ERROR);
                            return future;
                        }
                        if (!workerDestroyed) {
                            logger.warn(
                                    "WorkerThread timed out while cleaning, please resend request.");
                            future.complete(HttpResponseStatus.REQUEST_TIMEOUT);
                            return future;
                        }
                    }
                }
                if (!isStartup) {
                    SnapshotManager.getInstance().saveSnapshot();
                    isSnapshotSaved = true;
                }
                future.complete(HttpResponseStatus.OK);
            }
            if (!isStartup && !isSnapshotSaved) {
                SnapshotManager.getInstance().saveSnapshot();
            }
            return future;
        }
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
                            configManager.isDebug() ? port.get() : port.getAndIncrement(),
                            gpuId,
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
