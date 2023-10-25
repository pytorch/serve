package org.pytorch.serve.wlm;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.JobGroup;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SequenceBatchAggregator extends BatchAggregator {

    private static final Logger logger = LoggerFactory.getLogger(SequenceBatchAggregator.class);

    private final ConcurrentMap<String, Job> jobs;
    private LinkedHashSet<String> jobGroups;
    private final ExecutorService pollExecutors;

    public SequenceBatchAggregator(Model model) {
        super(model);
        jobs = new ConcurrentHashMap<>();
        if (model.isStateful()) {
            jobGroups = new LinkedHashSet<>(model.getBatchSize());
        }
        this.pollExecutors = Executors.newFixedThreadPool(model.getBatchSize());
    }

    private void pollJobGroup() throws InterruptedException {
        cleanAllJobGroups();
        LinkedHashSet<String> tmpJobGroups = new LinkedHashSet<>();
        if (jobGroups.size() == 0) {
            String jobGroupId =
                    model.getPendingJobGroups().poll(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
            jobGroups.add(jobGroupId);
            logger.debug("added jobGroup:{} in jobGroups", jobGroupId);
        }
        int quota = model.getBatchSize() - jobGroups.size();
        if (quota > 0 && model.getPendingJobGroups().size() > 0) {
            model.getPendingJobGroups().drainTo(tmpJobGroups, quota);
            jobGroups.addAll(tmpJobGroups);
        }
        if (!tmpJobGroups.isEmpty()) {
            logger.debug(
                    "added jobGroup:{} in jobGroups",
                    tmpJobGroups.stream().map(String::valueOf).collect(Collectors.joining(",")));
        }
    }

    public static <T> CompletableFuture<Void> allOfFutures(List<CompletableFuture<T>> futures) {
        CompletableFuture<Void> allDoneFuture;
        CompletableFuture<?>[] completableFuturesArray =
                futures.toArray(new CompletableFuture<?>[0]);
        allDoneFuture = CompletableFuture.allOf(completableFuturesArray);

        return allDoneFuture;
    }

    private void pollJobFromGroups() throws InterruptedException, ExecutionException {
        while (jobs.isEmpty()) {
            if (jobGroups.size() < model.getBatchSize()) {
                pollJobGroup();
            }

            List<CompletableFuture<Void>> futures = new ArrayList<>(jobGroups.size());
            for (String groupId : jobGroups) {
                JobGroup jobGroup = model.getJobGroup(groupId);
                futures.add(
                        CompletableFuture.runAsync(
                                () -> {
                                    Job job = jobGroup.pollJob((long) model.getMaxBatchDelay());
                                    if (job != null) {
                                        jobs.put(job.getJobId(), job);
                                    }
                                },
                                this.pollExecutors));
            }

            CompletableFuture<Void> allOf = allOfFutures(futures);
            allOf.get();

            if (jobs.isEmpty()) {
                cleanAllJobGroups();
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
            pollJobFromGroups();
        }
    }

    private void cleanJobGroup(String jobGroupId) {
        logger.debug("Clean jobGroup: {}", jobGroupId);
        if (jobGroupId != null) {
            JobGroup jobGroup = model.getJobGroup(jobGroupId);
            if (!jobGroup.groupHasNextInput() && jobGroup.size() == 0) {
                jobGroup.monitorShutdown();
                jobGroups.remove(jobGroupId);
                model.removeJobGroup(jobGroupId);
            }
        }
    }

    private void cleanAllJobGroups() {
        Iterator<String> it = jobGroups.iterator();
        while (it.hasNext()) {
            String jobGroupId = it.next();
            if (jobGroupId != null) {
                JobGroup jobGroup = model.getJobGroup(jobGroupId);
                if (!jobGroup.groupHasNextInput() && jobGroup.size() == 0) {
                    jobGroup.monitorShutdown();
                    it.remove();
                    model.removeJobGroup(jobGroupId);
                    logger.debug("Clean jobGroup={}", jobGroupId);
                }
            }
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

    public void shutdownExecutors() {
        this.pollExecutors.shutdown();
    }
}
