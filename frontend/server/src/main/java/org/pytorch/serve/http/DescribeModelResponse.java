package org.pytorch.serve.http;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class DescribeModelResponse {

    private String modelName;
    private String modelVersion;
    private String modelUrl;
    private String engine;
    private String runtime;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private String status;
    private boolean loadedAtStartup;

    private List<Worker> workers;
    private Metrics metrics;

    public DescribeModelResponse() {
        workers = new ArrayList<>();
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public boolean getLoadedAtStartup() {
        return loadedAtStartup;
    }

    public void setLoadedAtStartup(boolean loadedAtStartup) {
        this.loadedAtStartup = loadedAtStartup;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public String getModelUrl() {
        return modelUrl;
    }

    public void setModelUrl(String modelUrl) {
        this.modelUrl = modelUrl;
    }

    public String getEngine() {
        return engine;
    }

    public void setEngine(String engine) {
        this.engine = engine;
    }

    public String getRuntime() {
        return runtime;
    }

    public void setRuntime(String runtime) {
        this.runtime = runtime;
    }

    public int getMinWorkers() {
        return minWorkers;
    }

    public void setMinWorkers(int minWorkers) {
        this.minWorkers = minWorkers;
    }

    public int getMaxWorkers() {
        return maxWorkers;
    }

    public void setMaxWorkers(int maxWorkers) {
        this.maxWorkers = maxWorkers;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public List<Worker> getWorkers() {
        return workers;
    }

    public void setWorkers(List<Worker> workers) {
        this.workers = workers;
    }

    public void addWorker(
            String id, long startTime, boolean isRunning, int gpuId, long memoryUsage) {
        Worker worker = new Worker();
        worker.setId(id);
        worker.setStartTime(new Date(startTime));
        worker.setStatus(isRunning ? "READY" : "UNLOADING");
        worker.setGpu(gpuId >= 0);
        worker.setMemoryUsage(memoryUsage);
        workers.add(worker);
    }

    public Metrics getMetrics() {
        return metrics;
    }

    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    public static final class Worker {

        private String id;
        private Date startTime;
        private String status;
        private boolean gpu;
        private long memoryUsage;

        public Worker() {}

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public Date getStartTime() {
            return startTime;
        }

        public void setStartTime(Date startTime) {
            this.startTime = startTime;
        }

        public String getStatus() {
            return status;
        }

        public void setStatus(String status) {
            this.status = status;
        }

        public boolean isGpu() {
            return gpu;
        }

        public void setGpu(boolean gpu) {
            this.gpu = gpu;
        }

        public long getMemoryUsage() {
            return memoryUsage;
        }

        public void setMemoryUsage(long memoryUsage) {
            this.memoryUsage = memoryUsage;
        }
    }

    public static final class Metrics {

        private int rejectedRequests;
        private int waitingQueueSize;
        private int requests;

        public int getRejectedRequests() {
            return rejectedRequests;
        }

        public void setRejectedRequests(int rejectedRequests) {
            this.rejectedRequests = rejectedRequests;
        }

        public int getWaitingQueueSize() {
            return waitingQueueSize;
        }

        public void setWaitingQueueSize(int waitingQueueSize) {
            this.waitingQueueSize = waitingQueueSize;
        }

        public int getRequests() {
            return requests;
        }

        public void setRequests(int requests) {
            this.requests = requests;
        }
    }
}
