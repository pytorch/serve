package org.pytorch.serve.wlm;

import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.JobGroup;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SequenceBatchAggregator extends BatchAggregator {

    private static final Logger logger = LoggerFactory.getLogger(SequenceBatchAggregator.class);
    private ExecutorService pollExecutors;
    private LinkedBlockingDeque<String> eventJobGroupIds;
    private LinkedBlockingDeque<Job> jobQueue;
    private Thread eventDispatcher;
    private AtomicBoolean isPollJobGroup;
    private LinkedList<String> currentJobGroupIds;

    public SequenceBatchAggregator(Model model) {
        super(model);
        this.currentJobGroupIds = new LinkedList<>();
        this.pollExecutors = Executors.newFixedThreadPool(model.getBatchSize() + 1);
        this.jobQueue = new LinkedBlockingDeque<>();
        this.isPollJobGroup = new AtomicBoolean(false);
        this.eventJobGroupIds = new LinkedBlockingDeque<>();
        this.eventJobGroupIds.add("");
        this.eventDispatcher = new Thread(new EventDispatcher());
        this.eventDispatcher.start();
    }

    public void startEventDispatcher() {
        this.eventDispatcher.start();
    }

    public void stopEventDispatcher() {
        this.eventDispatcher.interrupt();
    }

    private void pollJobGroup() throws InterruptedException {
        if (isPollJobGroup.getAndSet(true)) {
            return;
        }
        LinkedHashSet<String> tmpJobGroups = new LinkedHashSet<>();
        String jobGroupId = model.getPendingJobGroups().poll(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
        if (jobGroupId != null) {
            addJobGroup(jobGroupId);

            int quota = model.getPendingJobGroups().size() / model.getMaxWorkers();
            if (quota > 0 && model.getPendingJobGroups().size() > 0) {
                model.getPendingJobGroups().drainTo(tmpJobGroups, quota);
            }

            for (String jGroupId : tmpJobGroups) {
                addJobGroup(jGroupId);
            }
        }
        isPollJobGroup.set(false);
    }

    private void pollInferJob() throws InterruptedException {
        Job job = jobQueue.poll(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
        if (job == null) {
            return;
        }
        jobs.put(job.getJobId(), job);
        if (job.getGroupId() != null) {
            currentJobGroupIds.add(job.getGroupId());
        }
        for (int i = 1; i < model.getBatchSize(); i++) {
            job = jobQueue.poll();
            if (job == null) {
                break;
            }
            jobs.put(job.getJobId(), job);
        }
    }

    /**
     * The priority of polling a batch P0: poll a job with one single management request. In this
     * case, the batchSize is 1. P1: poll jobs from job groups. In this case, the batch size is
     * equal to or less than the number of job groups storeed in this aggregator. P2: poll jobs from
     * the DEFAULT_DATA_QUEUE of this model.
     */
    @Override
    public void pollBatch(String threadName, WorkerState state)
            throws InterruptedException, ExecutionException {
        boolean pollMgmtJobStatus = false;
        if (jobs.isEmpty()) {
            pollMgmtJobStatus =
                    model.pollMgmtJob(
                            threadName,
                            (state == WorkerState.WORKER_MODEL_LOADED) ? 0 : Long.MAX_VALUE,
                            jobs);
        }

        if (!pollMgmtJobStatus && state == WorkerState.WORKER_MODEL_LOADED) {
            pollInferJob();
        }
    }

    private void cleanJobGroup(String jobGroupId) {
        logger.debug("Clean jobGroup: {}", jobGroupId);
        if (jobGroupId != null) {
            model.removeJobGroup(jobGroupId);
        }
    }

    @Override
    public void handleErrorJob(Job job) {
        if (job.getGroupId() == null) {
            model.addFirst(job);
        } else {
            logger.error(
                    "Failed to process requestId: {}, sequenceId: {}",
                    job.getPayload().getRequestId(),
                    job.getGroupId());
        }
    }

    @Override
    public boolean sendResponse(ModelWorkerResponse message) {
        boolean jobDone = super.sendResponse(message);
        if (jobDone && !currentJobGroupIds.isEmpty()) {
            eventJobGroupIds.addAll(currentJobGroupIds);
            currentJobGroupIds.clear();
        }
        return jobDone;
    }

    @Override
    public void sendError(BaseModelRequest message, String error, int status) {
        super.sendError(message, error, status);
        if (!currentJobGroupIds.isEmpty()) {
            eventJobGroupIds.addAll(currentJobGroupIds);
            currentJobGroupIds.clear();
        }
    }

    @Override
    public void cleanJobs() {
        super.cleanJobs();
        if (!currentJobGroupIds.isEmpty()) {
            eventJobGroupIds.addAll(currentJobGroupIds);
            currentJobGroupIds.clear();
        }
    }

    public void shutdownExecutors() {
        this.pollExecutors.shutdown();
    }

    private void addJobGroup(String jobGroupId) {
        if (jobGroupId != null) {
            eventJobGroupIds.add(jobGroupId);
        }
    }

    class EventDispatcher implements Runnable {
        @Override
        public void run() {
            while (true) {
                try {
                    String jobGroupId =
                            eventJobGroupIds.poll(model.getMaxBatchDelay(), TimeUnit.MILLISECONDS);
                    if (jobGroupId == null || jobGroupId.isEmpty()) {
                        CompletableFuture.runAsync(
                                () -> {
                                    try {
                                        pollJobGroup();
                                    } catch (InterruptedException e) {
                                        logger.error("Failed to poll a job group", e);
                                    }
                                },
                                pollExecutors);
                    } else {

                        CompletableFuture.runAsync(
                                () -> {
                                    pollJobFromJobGroup(jobGroupId);
                                },
                                pollExecutors);
                    }
                } catch (InterruptedException e) {
                    logger.error("EventDispatcher failed to get jobGroup", e);
                }
            }
        }

        private void pollJobFromJobGroup(String jobGroupId) {
            // Poll a job from a jobGroup
            JobGroup jobGroup = model.getJobGroup(jobGroupId);
            Job job = jobGroup.pollJob(model.getSequenceMaxIdleMSec());
            if (job == null) {
                // JobGroup expired, clean it.
                cleanJobGroup(jobGroupId);
                // intent to add new job groups.
                eventJobGroupIds.add("");
            } else {
                jobQueue.add(job);
            }
        }
    }
}
