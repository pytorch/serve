package org.pytorch.serve.wlm;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.JobGroup;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.ModelInferenceRequest;
import org.pytorch.serve.util.messages.ModelLoadModelRequest;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.Predictions;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BatchAggregator {

    private static final Logger logger = LoggerFactory.getLogger(BatchAggregator.class);

    private final Model model;
    private final ConcurrentMap<String, Job> jobs;
    private LinkedHashSet<String> jobGroups;

    public BatchAggregator(Model model) {
        this.model = model;
        jobs = new ConcurrentHashMap<>();
        if (model.isStateful()) {
            jobGroups = new LinkedHashSet<>(model.getBatchSize());
        }
    }

    public BaseModelRequest getRequest(String threadName, WorkerState state)
            throws InterruptedException {
        jobs.clear();

        ModelInferenceRequest req = new ModelInferenceRequest(model.getModelName());

        pollBatch(threadName, state);

        if (model.isUseJobTicket() && jobs.isEmpty()) {
            model.decNumJobTickets();
            return req;
        }

        for (Job j : jobs.values()) {
            if (j.isControlCmd()) {
                if (jobs.size() > 1) {
                    throw new IllegalStateException(
                            "Received more than 1 control command. "
                                    + "Control messages should be processed/retrieved one at a time.");
                }
                RequestInput input = j.getPayload();
                int gpuId = -1;
                String gpu = input.getStringParameter("gpu");
                if (gpu != null) {
                    gpuId = Integer.parseInt(gpu);
                }
                return new ModelLoadModelRequest(model, gpuId);
            } else {
                WorkerCommands workerCmd = j.getCmd();
                if (workerCmd == WorkerCommands.STREAMPREDICT || workerCmd == WorkerCommands.STREAMPREDICT2) {
                    req.setCommand(workerCmd);
                }
                j.setScheduled();
                req.addRequest(j.getPayload());
            }
        }
        return req;
    }

    /**
     * @param message: a response of a batch inference requests
     * @return - true: either a non-stream response or last stream response is sent - false: a
     * stream response (not include the last stream) is sent
     */
    public boolean sendResponse(ModelWorkerResponse message) {
        boolean jobDone = true;
        // TODO: Handle prediction level code
        if (message.getCode() == 200) {
            if (jobs.isEmpty()) {
                // this is from initial load.
                return true;
            }
            for (Predictions prediction : message.getPredictions()) {
                String jobId = prediction.getRequestId();
                Job job = jobs.get(jobId);

                if (job == null) {
                    throw new IllegalStateException(
                            "Unexpected job in sendResponse() with 200 status code: " + jobId);
                }
                if (jobDone) {
                    String streamNext =
                            prediction
                                    .getHeaders()
                                    .get(
                                            org.pytorch.serve.util.messages.RequestInput
                                                    .TS_STREAM_NEXT);
                    if (streamNext != null && streamNext.equals("true")) {
                        jobDone = false;
                    }
                }
                if (job.getPayload().getClientExpireTS() > System.currentTimeMillis()) {
                    job.response(
                            prediction.getResp(),
                            prediction.getContentType(),
                            prediction.getStatusCode(),
                            prediction.getReasonPhrase(),
                            prediction.getHeaders());
                } else {
                    logger.warn(
                            "Drop response for inference request {} due to client timeout",
                            job.getPayload().getRequestId());
                }
                if (jobDone) {
                    cleanJobGroup(job.getGroupId());
                }
            }
        } else {
            for (Map.Entry<String, Job> j : jobs.entrySet()) {
                if (j.getValue() == null) {
                    throw new IllegalStateException(
                            "Unexpected job in sendResponse() with non 200 status code: "
                                    + j.getKey());
                }
                Job job = j.getValue();
                if (job.getPayload().getClientExpireTS() > System.currentTimeMillis()) {
                    job.sendError(message.getCode(), message.getMessage());
                } else {
                    logger.warn(
                            "Drop error response for inference request {} due to client timeout",
                            job.getPayload().getRequestId());
                }
                cleanJobGroup(job.getGroupId());
            }
        }
        if (jobDone) {
            jobs.clear();
        }
        return jobDone;
    }

    public void sendError(BaseModelRequest message, String error, int status) {
        if (message instanceof ModelLoadModelRequest) {
            logger.warn("Load model failed: {}, error: {}", message.getModelName(), error);
            return;
        }

        if (message != null) {
            ModelInferenceRequest msg = (ModelInferenceRequest) message;
            for (RequestInput req : msg.getRequestBatch()) {
                String requestId = req.getRequestId();
                Job job = jobs.remove(requestId);
                if (job == null) {
                    logger.error("Unexpected job in sendError(): " + requestId);
                } else {
                    job.sendError(status, error);
                    cleanJobGroup(job.getGroupId());
                }
            }
            if (!jobs.isEmpty()) {
                jobs.clear();
                logger.error("Not all jobs got an error response.");
            }
        } else {
            // Send the error message to all the jobs
            for (Map.Entry<String, Job> j : jobs.entrySet()) {
                String jobsId = j.getValue().getJobId();
                Job job = jobs.get(jobsId);

                if (job.isControlCmd()) {
                    job.sendError(status, error);
                } else {
                    // Data message can be handled by other workers.
                    // If batch has gone past its batch max delay timer?
                    if (job.getGroupId() == null) {
                        model.addFirst(job);
                    } else {
                        logger.error("Failed to process requestId: {}, sequenceId: {}",
                                job.getPayload().getRequestId(), job.getGroupId());
                        cleanJobGroup(job.getGroupId());
                    }
                }
            }
        }
        jobs.clear();
    }

    private void pollJobGroup() throws InterruptedException {
        cleanJobGroup();
        LinkedHashSet<String> tmpJobGroups = new LinkedHashSet<>();
        if (jobGroups.size() == 0) {
            jobGroups.add(model.getPendingJobGroups().poll(Long.MAX_VALUE, TimeUnit.MILLISECONDS));
        }
        int quota = model.getBatchSize() - jobGroups.size();
        if (quota > 0 && model.getPendingJobGroups().size() > 0) {
            model.getPendingJobGroups().drainTo(tmpJobGroups, quota);
            jobGroups.addAll(tmpJobGroups);
        }
    }

    public void pollJobFromGroups() throws InterruptedException {
        if (jobGroups.size() < model.getBatchSize()) {
            pollJobGroup();
        }

        List<CompletableFuture<Void>> futures = new ArrayList<>(jobGroups.size());
        for (String groupId : jobGroups) {
            JobGroup jobGroup = model.getJobGroup(groupId);
            futures.add(CompletableFuture.supplyAsync(
                    () -> jobGroup.pollJob(model.getSequenceMaxIdleMSec()))
                    .thenAccept(job -> {
                        if (job != null) {
                            jobs.put(job.getJobId(), job);
                        }
                    }));
        }
        futures.stream().map(CompletableFuture::join);
    }

    /**
     * The priority of polling a batch
     * P0: poll a job with one single management request. In this case, the batchSize is 1.
     * P1: poll jobs from job groups. In this case, the batch size is equal to or less than the number
     *     of job groups storeed in this aggregator.
     * P2: poll jobs from the DEFAULT_DATA_QUEUE of this model.
     */
    private void pollBatch(String threadName, WorkerState state) throws InterruptedException {
        if (model.pollMgmtJob(
                threadName,
                (state == WorkerState.WORKER_MODEL_LOADED) ? 0 : Long.MAX_VALUE,
                jobs)) {
        } else if (model.isStateful()) {
            pollJobFromGroups();
        } else {
            model.pollInferJob(jobs);
        }
    }

    private void cleanJobGroup(String jobGroupId) {
        if (jobGroupId != null) {
            JobGroup jobGroup = model.getJobGroup(jobGroupId);
            if (!jobGroup.groupHasNextInput() && jobGroup.size() == 0) {
                jobGroups.remove(jobGroupId);
                model.removeJobGroup(jobGroupId);
            }
        }
    }

    private void cleanJobGroup() {
        Iterator<String> it = jobGroups.iterator();
        while (it.hasNext()) {
            String jobGroupId = it.next();
            JobGroup jobGroup = model.getJobGroup(it.next());
            if (!jobGroup.groupHasNextInput() && jobGroup.size() == 0) {
                jobGroups.remove(jobGroupId);
                model.removeJobGroup(jobGroupId);
            }
        }
    }
}
