package org.pytorch.serve.wlm;

import com.google.gson.JsonObject;
import java.io.File;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import org.apache.commons.io.FilenameUtils;
import org.pytorch.serve.archive.ModelArchive;
import org.pytorch.serve.util.ConfigManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Model {

    public static final String DEFAULT_DATA_QUEUE = "DATA_QUEUE";

    public static final String MIN_WORKERS = "minWorkers";
    public static final String MAX_WORKERS = "maxWorkers";
    public static final String BATCH_SIZE = "batchSize";
    public static final String MAX_BATCH_DELAY = "maxBatchDelay";
    public static final String RESPONSE_TIMEOUT = "responseTimeout";
    public static final String DEFAULT_VERSION = "defaultVersion";
    public static final String MAR_NAME = "marName";

    private static final Logger logger = LoggerFactory.getLogger(Model.class);

    private ModelArchive modelArchive;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private ReentrantLock lock;
    private int responseTimeout;
    private ModelVersionName modelVersionName;

    // Total number of subsequent inference request failures
    private AtomicInteger failedInfReqs;

    // Per worker thread job queue. This separates out the control queue from data queue
    private ConcurrentMap<String, LinkedBlockingDeque<Job>> jobsDb;

    public Model(ModelArchive modelArchive, int queueSize) {
        this.modelArchive = modelArchive;
        batchSize = 1;
        maxBatchDelay = 100;
        jobsDb = new ConcurrentHashMap<>();
        // Always have a queue for data
        jobsDb.putIfAbsent(DEFAULT_DATA_QUEUE, new LinkedBlockingDeque<>(queueSize));
        failedInfReqs = new AtomicInteger(0);
        lock = new ReentrantLock();
        modelVersionName =
                new ModelVersionName(
                        this.modelArchive.getModelName(), this.modelArchive.getModelVersion());
    }

    public JsonObject getModelState(boolean isDefaultVersion) {

        JsonObject modelInfo = new JsonObject();
        modelInfo.addProperty(DEFAULT_VERSION, isDefaultVersion);
        modelInfo.addProperty(MAR_NAME, FilenameUtils.getName(getModelUrl()));
        modelInfo.addProperty(MIN_WORKERS, getMinWorkers());
        modelInfo.addProperty(MAX_WORKERS, getMaxWorkers());
        modelInfo.addProperty(BATCH_SIZE, getBatchSize());
        modelInfo.addProperty(MAX_BATCH_DELAY, getMaxBatchDelay());
        modelInfo.addProperty(RESPONSE_TIMEOUT, getResponseTimeout());

        return modelInfo;
    }

    public void setModelState(JsonObject modelInfo) {
        minWorkers = modelInfo.get(MIN_WORKERS).getAsInt();
        maxWorkers = modelInfo.get(MAX_WORKERS).getAsInt();
        maxBatchDelay = modelInfo.get(MAX_BATCH_DELAY).getAsInt();
        responseTimeout = modelInfo.get(RESPONSE_TIMEOUT).getAsInt();
        batchSize = modelInfo.get(BATCH_SIZE).getAsInt();
    }

    public String getModelName() {
        return modelArchive.getModelName();
    }

    public ModelVersionName getModelVersionName() {
        return modelVersionName;
    }

    public String getVersion() {
        return modelArchive.getModelVersion();
    }

    public File getModelDir() {
        return modelArchive.getModelDir();
    }

    public String getModelUrl() {
        return modelArchive.getUrl();
    }

    public ModelArchive getModelArchive() {
        return modelArchive;
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

    public void addJob(String threadId, Job job) {
        LinkedBlockingDeque<Job> blockingDeque = jobsDb.get(threadId);
        if (blockingDeque == null) {
            blockingDeque = new LinkedBlockingDeque<>();
            jobsDb.put(threadId, blockingDeque);
        }
        blockingDeque.offer(job);
    }

    public void removeJobQueue(String threadId) {
        if (!threadId.equals(DEFAULT_DATA_QUEUE)) {
            jobsDb.remove(threadId);
        }
    }

    public boolean addJob(Job job) {
        return jobsDb.get(DEFAULT_DATA_QUEUE).offer(job);
    }

    public void addFirst(Job job) {
        jobsDb.get(DEFAULT_DATA_QUEUE).addFirst(job);
    }

    public void pollBatch(String threadId, long waitTime, Map<String, Job> jobsRepo)
            throws InterruptedException {
        if (jobsRepo == null || threadId == null || threadId.isEmpty()) {
            throw new IllegalArgumentException("Invalid input given provided");
        }

        if (!jobsRepo.isEmpty()) {
            throw new IllegalArgumentException(
                    "The jobs repo provided contains stale jobs. Clear them!!");
        }

        LinkedBlockingDeque<Job> jobsQueue = jobsDb.get(threadId);
        if (jobsQueue != null && !jobsQueue.isEmpty()) {
            Job j = jobsQueue.poll(waitTime, TimeUnit.MILLISECONDS);
            if (j != null) {
                jobsRepo.put(j.getJobId(), j);
                return;
            }
        }

        try {
            lock.lockInterruptibly();
            long maxDelay = maxBatchDelay;
            jobsQueue = jobsDb.get(DEFAULT_DATA_QUEUE);

            Job j = jobsQueue.poll(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
            logger.trace("get first job: {}", Objects.requireNonNull(j).getJobId());

            jobsRepo.put(j.getJobId(), j);
            long begin = System.currentTimeMillis();
            for (int i = 0; i < batchSize - 1; ++i) {
                j = jobsQueue.poll(maxDelay, TimeUnit.MILLISECONDS);
                if (j == null) {
                    break;
                }
                long end = System.currentTimeMillis();
                maxDelay -= end - begin;
                begin = end;
                jobsRepo.put(j.getJobId(), j);
                if (maxDelay <= 0) {
                    break;
                }
            }
            logger.trace("sending jobs, size: {}", jobsRepo.size());
        } finally {
            if (lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }

    public int incrFailedInfReqs() {
        return failedInfReqs.incrementAndGet();
    }

    public void resetFailedInfReqs() {
        failedInfReqs.set(0);
    }

    public int getResponseTimeout() {
        return ConfigManager.getInstance().isDebug() ? Integer.MAX_VALUE : responseTimeout;
    }

    public void setResponseTimeout(int responseTimeout) {
        this.responseTimeout = responseTimeout;
    }
}
