package org.pytorch.serve.wlm;

import java.time.Duration;
import java.time.Instant;
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

public class SequenceBatching extends BatchAggregator {

    private static final Logger logger = LoggerFactory.getLogger(SequenceBatching.class);
    private ExecutorService pollExecutors;
    /**
     * eventJobGroupIds is an queue in EventDispatcher. It's item has 2 cases. 1) empty string:
     * trigger EventDispatcher to fetch new job groups. 2) job group id: trigger EventDispatcher to
     * fetch a new job from this jobGroup.
     */
    protected LinkedBlockingDeque<String> eventJobGroupIds;
    // A queue holds jobs ready for this aggregator to add into a batch. Each job of this queue is
    // from distinct jobGroup. jobs
    protected LinkedBlockingDeque<Job> jobsQueue;
    private Thread eventDispatcher;
    private AtomicBoolean isPollJobGroup;
    // A list of jobGroupIds which are added into current batch. These jobGroupIds need to be added
    // back to eventJobGroupIds once their jobs are processed by a batch.
    protected LinkedList<String> currentJobGroupIds;
    private int localCapacity;
    private AtomicBoolean running = new AtomicBoolean(true);

    public SequenceBatching(Model model) {
        super(model);
        this.localCapacity = Math.max(1, model.getMaxNumSequence() / model.getMinWorkers());
        this.currentJobGroupIds = new LinkedList<>();
        this.pollExecutors = Executors.newFixedThreadPool(model.getBatchSize() + 1);
        this.jobsQueue = new LinkedBlockingDeque<>();
        this.isPollJobGroup = new AtomicBoolean(false);
        this.eventJobGroupIds = new LinkedBlockingDeque<>();
        this.eventJobGroupIds.add("");
        this.eventDispatcher = new Thread(new EventDispatcher());
        this.eventDispatcher.start();
    }

    @Override
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

            int quota =
                    Math.min(
                            this.localCapacity - jobsQueue.size(),
                            Math.max(
                                    1, model.getPendingJobGroups().size() / model.getMaxWorkers()));
            if (quota > 0 && model.getPendingJobGroups().size() > 0) {
                model.getPendingJobGroups().drainTo(tmpJobGroups, quota);
            }

            for (String jGroupId : tmpJobGroups) {
                addJobGroup(jGroupId);
            }
        }
        isPollJobGroup.set(false);
    }

    protected void pollInferJob() throws InterruptedException {
        model.pollInferJob(jobs, model.getBatchSize(), jobsQueue);

        for (Job job : jobs.values()) {
            if (job.getGroupId() != null) {
                currentJobGroupIds.add(job.getGroupId());
            }
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

    protected void cleanJobGroup(String jobGroupId) {
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

    @Override
    public void shutdown() {
        this.setRunning(false);
        this.shutdownExecutors();
        this.stopEventDispatcher();
    }

    public void shutdownExecutors() {
        this.pollExecutors.shutdown();
    }

    private void addJobGroup(String jobGroupId) {
        if (jobGroupId != null) {
            eventJobGroupIds.add(jobGroupId);
        }
    }

    public void setRunning(boolean running) {
        this.running.set(running);
    }

    class EventDispatcher implements Runnable {
        @Override
        public void run() {
            while (running.get()) {
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
                    if (running.get()) {
                        logger.error("EventDispatcher failed to get jobGroup", e);
                    }
                }
            }
        }

        private void pollJobFromJobGroup(String jobGroupId) {
            // Poll a job from a jobGroup
            JobGroup jobGroup = model.getJobGroup(jobGroupId);
            Job job = null;
            AtomicBoolean isPolling = jobGroup.getPolling();
            if (!jobGroup.isFinished()) {
                if (!isPolling.getAndSet(true)) {
                    job = jobGroup.pollJob(getPollJobGroupTimeoutMSec(jobGroup));
                    isPolling.set(false);
                } else {
                    return;
                }
            }
            if (job == null) {
                // JobGroup expired, clean it.
                cleanJobGroup(jobGroupId);
                // intent to add new job groups.
                eventJobGroupIds.add("");
            } else {
                jobsQueue.add(job);
            }
        }

        private long getPollJobGroupTimeoutMSec(JobGroup jobGroup) {
            long pollTimeout = 0;
            Instant currentTimestamp = Instant.now();
            Instant expiryTimestamp = jobGroup.getExpiryTimestamp();

            if (expiryTimestamp == Instant.MAX) {
                pollTimeout = model.getSequenceMaxIdleMSec();
            } else if (currentTimestamp.isBefore(expiryTimestamp)) {
                long remainingPollDuration =
                        Duration.between(currentTimestamp, expiryTimestamp).toMillis();
                pollTimeout = Math.min(model.getSequenceMaxIdleMSec(), remainingPollDuration);
            }

            return pollTimeout;
        }
    }
}
