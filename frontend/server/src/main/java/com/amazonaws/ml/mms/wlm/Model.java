/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.ml.mms.wlm;

import com.amazonaws.ml.mms.archive.ModelArchive;
import com.amazonaws.ml.mms.util.ConfigManager;
import java.io.File;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Model {

    public static final String DEFAULT_DATA_QUEUE = "DATA_QUEUE";
    private static final Logger logger = LoggerFactory.getLogger(Model.class);

    private ModelArchive modelArchive;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private ReentrantLock lock;
    private int responseTimeout;

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
    }

    public String getModelName() {
        return modelArchive.getModelName();
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
            lock.unlock();
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
