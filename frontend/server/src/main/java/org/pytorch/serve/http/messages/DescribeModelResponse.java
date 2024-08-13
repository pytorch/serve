package org.pytorch.serve.http.messages;

import com.google.gson.JsonObject;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import org.pytorch.serve.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DescribeModelResponse {
    private static final Logger logger = LoggerFactory.getLogger(DescribeModelResponse.class);

    private String modelName;
    private String modelVersion;
    private String modelUrl;
    private String engine;
    private String runtime;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private int responseTimeout;
    private int startupTimeout;
    private long maxRetryTimeoutInSec;
    private long clientTimeoutInMills;
    private String parallelType;
    private int parallelLevel;
    private String deviceType;
    private List<Integer> deviceIds;
    private boolean continuousBatching;
    private boolean useJobTicket;
    private boolean useVenv;
    private boolean stateful;
    private long sequenceMaxIdleMSec;
    private long sequenceTimeoutMSec;
    private int maxNumSequence;
    private int maxSequenceJobQueueSize;
    private String status;
    private boolean loadedAtStartup;

    private List<Worker> workers;
    private Metrics metrics;
    private JobQueueStatus jobQueueStatus;
    private JsonObject customizedMetadata;

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

    public int getResponseTimeout() {
        return responseTimeout;
    }

    public int getStartupTimeout() {
        return startupTimeout;
    }

    public void setResponseTimeout(int responseTimeout) {
        this.responseTimeout = responseTimeout;
    }

    public void setStartupTimeout(int startupTimeout) {
        this.startupTimeout = startupTimeout;
    }

    public long getMaxRetryTimeoutInSec() {
        return maxRetryTimeoutInSec;
    }

    public void setMaxRetryTimeoutInSec(long maxRetryTimeoutInSec) {
        this.maxRetryTimeoutInSec = maxRetryTimeoutInSec;
    }

    public long getClientTimeoutInMills() {
        return clientTimeoutInMills;
    }

    public void setClientTimeoutInMills(long clientTimeoutInMills) {
        this.clientTimeoutInMills = clientTimeoutInMills;
    }

    public String getParallelType() {
        return parallelType;
    }

    public void setParallelType(String parallelType) {
        this.parallelType = parallelType;
    }

    public int getParallelLevel() {
        return parallelLevel;
    }

    public void setParallelLevel(int parallelLevel) {
        this.parallelLevel = parallelLevel;
    }

    public String getDeviceType() {
        return deviceType;
    }

    public void setDeviceType(String deviceType) {
        this.deviceType = deviceType;
    }

    public List<Integer> getDeviceIds() {
        return deviceIds;
    }

    public void setDeviceIds(List<Integer> deviceIds) {
        this.deviceIds = deviceIds;
    }

    public boolean getContinuousBatching() {
        return continuousBatching;
    }

    public void setContinuousBatching(boolean continuousBatching) {
        this.continuousBatching = continuousBatching;
    }

    public boolean getUseJobTicket() {
        return useJobTicket;
    }

    public void setUseJobTicket(boolean useJobTicket) {
        this.useJobTicket = useJobTicket;
    }

    public boolean getUseVenv() {
        return useVenv;
    }

    public void setUseVenv(boolean useVenv) {
        this.useVenv = useVenv;
    }

    public boolean getStateful() {
        return stateful;
    }

    public void setStateful(boolean stateful) {
        this.stateful = stateful;
    }

    public long getSequenceMaxIdleMSec() {
        return sequenceMaxIdleMSec;
    }

    public void setSequenceMaxIdleMSec(long sequenceMaxIdleMSec) {
        this.sequenceMaxIdleMSec = sequenceMaxIdleMSec;
    }

    public long getSequenceTimeoutMSec() {
        return sequenceTimeoutMSec;
    }

    public void setSequenceTimeoutMSec(long sequenceTimeoutMSec) {
        this.sequenceTimeoutMSec = sequenceTimeoutMSec;
    }

    public int getMaxNumSequence() {
        return maxNumSequence;
    }

    public void setMaxNumSequence(int maxNumSequence) {
        this.maxNumSequence = maxNumSequence;
    }

    public int getMaxSequenceJobQueueSize() {
        return maxSequenceJobQueueSize;
    }

    public void setMaxSequenceJobQueueSize(int maxSequenceJobQueueSize) {
        this.maxSequenceJobQueueSize = maxSequenceJobQueueSize;
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
            String id,
            long startTime,
            boolean isRunning,
            int gpuId,
            long memoryUsage,
            int pid,
            String gpuUsage) {
        Worker worker = new Worker();
        worker.setId(id);
        worker.setStartTime(new Date(startTime));
        worker.setStatus(isRunning ? "READY" : "UNLOADING");
        worker.setMemoryUsage(memoryUsage);
        worker.setPid(pid);
        worker.setGpu(gpuId >= 0);
        worker.setGpuUsage(gpuUsage);
        workers.add(worker);
    }

    public Metrics getMetrics() {
        return metrics;
    }

    public void setMetrics(Metrics metrics) {
        this.metrics = metrics;
    }

    public JobQueueStatus getJobQueueStatus() {
        return jobQueueStatus;
    }

    public void setJobQueueStatus(JobQueueStatus jobQueueStatus) {
        this.jobQueueStatus = jobQueueStatus;
    }

    public void setCustomizedMetadata(byte[] customizedMetadata) {
        String stringMetadata = new String(customizedMetadata, Charset.forName("UTF-8"));
        try {
            this.customizedMetadata = JsonUtils.GSON.fromJson(stringMetadata, JsonObject.class);
        } catch (com.google.gson.JsonSyntaxException ex) {
            logger.warn("Customized metadata should be a dictionary.");
            this.customizedMetadata = new JsonObject();
        }
    }

    public JsonObject getCustomizedMetadata() {
        return customizedMetadata;
    }

    public static final class Worker {

        private String id;
        private Date startTime;
        private String status;
        private long memoryUsage;
        private int pid;
        private boolean gpu;
        private String gpuUsage;

        public Worker() {}

        public String getGpuUsage() {
            return gpuUsage;
        }

        public void setGpuUsage(String gpuUsage) {
            this.gpuUsage = gpuUsage;
        }

        public int getPid() {
            return pid;
        }

        public void setPid(int pid) {
            this.pid = pid;
        }

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

    public static final class JobQueueStatus {

        private int remainingCapacity;
        private int pendingRequests;

        public int getRemainingCapacity() {
            return remainingCapacity;
        }

        public void setRemainingCapacity(int remainingCapacity) {
            this.remainingCapacity = remainingCapacity;
        }

        public int getPendingRequests() {
            return pendingRequests;
        }

        public void setPendingRequests(int pendingRequests) {
            this.pendingRequests = pendingRequests;
        }
    }
}
