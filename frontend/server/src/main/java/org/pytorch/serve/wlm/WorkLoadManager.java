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

    private static final Logger logger = LoggerFactory.getLogger(WorkLoadManager.class);
    private ExecutorService threadPool;
    private ConcurrentHashMap<ModelVersionName, List<WorkerThread>> workers;
    private ConfigManager configManager;
    private EventLoopGroup backendGroup;
    private AtomicInteger port;
    private AtomicInteger distributionPort;
    private AtomicInteger gpuCounter;

    public WorkLoadManager(ConfigManager configManager, EventLoopGroup backendGroup) {
        this.configManager = configManager;
        this.backendGroup = backendGroup;
        this.port = new AtomicInteger(configManager.getInitialWorkerPort());
        this.distributionPort = new AtomicInteger(configManager.getInitialDistributionPort());
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

    public int getNumHealthyWorkers(ModelVersionName modelVersionName) {
        int numHealthy = 0;
        List<WorkerThread> threads = workers.getOrDefault(modelVersionName, null);

        if (threads != null) {
            for (WorkerThread thread : threads) {
                if (thread.isHealthy()) {
                    numHealthy += 1;
                }
            }
        }

        return numHealthy;
    }

    /**
     * Checks if cpu_launcher is enabled and currentWorkers > 0 (i.e., not initializing workers).
     * Workers are restarted so that when dynamically scaling the number of workers, cores that were
     * pinned to killed workers by the launcher are not left unutilizied. If isRestart, workers are
     * restarted to re-distribute cores that were pinned to killed workers to the remaining, alive
     * workers.
     */
    public boolean isLauncherRestartWorkers(int currentWorkers) {
        return configManager.isCPULauncherEnabled() && currentWorkers > 0;
    }

    public CompletableFuture<Integer> modelChanged(
            Model model, boolean isStartup, boolean isCleanUp) {
        synchronized (model.getModelVersionName()) {
            boolean isSnapshotSaved = false;
            CompletableFuture<Integer> future = new CompletableFuture<>();
            int minWorker = model.getMinWorkers();
            int maxWorker = model.getMaxWorkers();
            // Sets restartNumWorkers to the updated minWorker after scale up/down
            int restartNumWorkers = minWorker;
            List<WorkerThread> threads;
            if (minWorker == 0) {
                threads = workers.remove(model.getModelVersionName());
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
            boolean isRestartWorkers = isLauncherRestartWorkers(currentWorkers);

            if (isRestartWorkers) {
                logger.warn(
                        "removing {} current thread(s) prior to restarting {} thread(s)",
                        currentWorkers,
                        minWorker);
                // By setting maxWorker and minWorker to 0, removes all currentWorkers
                maxWorker = 0;
                minWorker = 0;
            }

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
                        try {
                            String cmd = String.format(OSUtils.getKillCmd(), workerProcess.pid());
                            Process workerKillProcess = Runtime.getRuntime().exec(cmd, null, null);
                            workerDestroyed =
                                    workerKillProcess.waitFor(
                                            configManager.getUnregisterModelTimeout(),
                                            TimeUnit.SECONDS);
                        } catch (InterruptedException | IOException e) {
                            logger.warn(
                                    "WorkerThread interrupted during waitFor, possible async resource cleanup.");
                            future.complete(HttpURLConnection.HTTP_INTERNAL_ERROR);
                            return future;
                        }
                        if (!workerDestroyed) {
                            logger.warn(
                                    "WorkerThread timed out while cleaning, please resend request.");
                            future.complete(HttpURLConnection.HTTP_CLIENT_TIMEOUT);
                            return future;
                        }
                    }
                }
                if (!isStartup && !isCleanUp && !model.isWorkflowModel()) {
                    SnapshotManager.getInstance().saveSnapshot();
                    isSnapshotSaved = true;
                }
                future.complete(HttpURLConnection.HTTP_OK);
            }

            // After removing all currentWorkers, add back (i.e., restart) restartNumWorkers
            if (isRestartWorkers) {
                logger.warn("restarting {} thread(s)", restartNumWorkers);
                addThreads(threads, model, restartNumWorkers, future);
            }

            if (!isStartup && !isSnapshotSaved && !isCleanUp && !model.isWorkflowModel()) {
                SnapshotManager.getInstance().saveSnapshot();
            }
            return future;
        }
    }

    private void addThreads(
            List<WorkerThread> threads, Model model, int count, CompletableFuture<Integer> future) {
        WorkerStateListener listener = new WorkerStateListener(future, count);
        int maxGpu = model.getNumCores();
        int stride = model.getParallelLevel() > 0 ? model.getParallelLevel() : 1;
        for (int i = 0; i < count; ++i) {
            int gpuId = -1;

            if (maxGpu > 0) {
                if (model.isHasCfgDeviceIds() || model.getParallelLevel() > 0) {
                    gpuId =
                            model.getGpuCounter()
                                    .getAndAccumulate(
                                            stride, (prev, myStride) -> (prev + myStride) % maxGpu);
                    if (model.getParallelLevel() == 0) {
                        gpuId = model.getDeviceIds().get(gpuId);
                    }
                } else {
                    gpuId =
                            gpuCounter.accumulateAndGet(
                                    maxGpu, (prev, maxGpuId) -> ++prev % maxGpuId);
                }
            }

            BatchAggregator aggregator;

            if (model.isSequenceBatching() && model.isContinuousBatching()) {
                aggregator = new SequenceContinuousBatching(model);
            } else if (model.isSequenceBatching()) {
                aggregator = new SequenceBatching(model);
            } else if (model.isContinuousBatching()) {
                aggregator = new ContinuousBatching(model);
            } else {
                aggregator = new BatchAggregator(model);
            }

            int currentPort =
                    model.getParallelLevel() > 0
                            ? configManager.isDebug()
                                    ? distributionPort.get()
                                    : distributionPort.getAndAdd(model.getParallelLevel())
                            : configManager.isDebug() ? port.get() : port.getAndIncrement();
            WorkerThread thread =
                    new WorkerThread(
                            configManager,
                            backendGroup,
                            currentPort,
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
