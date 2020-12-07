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
import java.util.concurrent.atomic.AtomicInteger;
import org.pytorch.serve.snapshot.SnapshotManager;
import org.pytorch.serve.util.ConfigManager;
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
        this.port = new AtomicInteger(configManager.getIniitialWorkerPort());
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

    public CompletableFuture<HttpResponseStatus> modelChanged(
            Model model, boolean isStartup, boolean isCleanUp) {
        synchronized (model.getModelVersionName()) {
            boolean isSnapshotSaved = false;
            CompletableFuture<HttpResponseStatus> future = new CompletableFuture<>();
            int minWorker = model.getMinWorkers();
            int maxWorker = model.getMaxWorkers();
            WorkerManagerThread workerManagerThread;
            List<WorkerThread> threads;
            if (minWorker == 0) {
                workerManagers.remove(model.getModelVersionName());
                threads = workers.remove(model.getModelVersionName());
                if (threads == null) {
                    future.complete(HttpResponseStatus.OK);
                    if (!isStartup && !isCleanUp) {
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
                if(currentWorkers == 0 ) {
                    workerManagerThread = workerManagers
                            .computeIfAbsent(
                                    model.getModelVersionName(),
                                    k -> addWorkerManagerThread(model, future));
                }
                addThreads(threads, model, minWorker - currentWorkers, future);
            } else {
                for (int i = currentWorkers - 1; i >= maxWorker; --i) {
                    WorkerThread thread = threads.remove(i);
                    workerManagerThread = workerManagers.get(model.getModelVersionName());
                    workerManagerThread.scaleDown(thread.getLifeCycle().getPort());
                    thread.shutdown();

                    if(threads.size() == 0){
                        workerManagerThread.getLifeCycle().exit();
                        workerManagerThread.shutdown();
                        workerManagers.remove(workerManagerThread);
                    }
                }
                if (!isStartup && !isCleanUp) {
                    SnapshotManager.getInstance().saveSnapshot();
                    isSnapshotSaved = true;
                }
                future.complete(HttpResponseStatus.OK);
            }
            if (!isStartup && !isSnapshotSaved && !isCleanUp) {
                SnapshotManager.getInstance().saveSnapshot();
            }
            return future;
        }
    }

    private WorkerManagerThread addWorkerManagerThread(
            Model model,
            CompletableFuture<HttpResponseStatus> future) {

        WorkerManagerStateListener listener = new WorkerManagerStateListener(future, 1);
        BatchAggregator aggregator = new BatchAggregator(model);

        WorkerManagerThread thread =
                new WorkerManagerThread(
                        configManager,
                        backendGroup,
                        configManager.isDebug() ? port.get() : port.getAndIncrement(),
                        model,
                        aggregator,
                        listener);
        threadPool.submit(thread);
        return thread;
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

            int portNum = port.incrementAndGet();
            workerManagers.get(model.getModelVersionName()).scaleUp(portNum);

            BatchAggregator aggregator = new BatchAggregator(model);
            WorkerThread thread =
                    new WorkerThread(
                            configManager,
                            backendGroup,
                            portNum,
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
