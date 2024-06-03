package org.pytorch.serve.wlm;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.ModelInferenceRequest;
import org.pytorch.serve.util.messages.ModelLoadModelRequest;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.Predictions;
import org.pytorch.serve.util.messages.RequestInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BatchAggregator {

    private static final Logger logger = LoggerFactory.getLogger(BatchAggregator.class);

    protected Model model;
    protected Map<String, Job> jobs;

    public BatchAggregator() {}

    public BatchAggregator(Model model) {
        this.model = model;
        jobs = new LinkedHashMap<>();
    }

    public BaseModelRequest getRequest(String threadName, WorkerState state)
            throws InterruptedException, ExecutionException {
        cleanJobs();

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
                req.setCommand(j.getCmd());
                j.setScheduled();
                req.addRequest(j.getPayload());
            }
        }
        return req;
    }

    /**
     * @param message: a response of a batch inference requests
     * @return - true: either a non-stream response or last stream response is sent - false: a
     *     stream response (not include the last stream) is sent
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
                    if ("true".equals(streamNext)) {
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
            }
        }
        if (jobDone) {
            cleanJobs();
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
                }
            }
            if (!jobs.isEmpty()) {
                cleanJobs();
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
                    handleErrorJob(job);
                }
            }
        }
        cleanJobs();
    }

    public void cleanJobs() {
        if (jobs != null) {
            jobs.clear();
        }
    }

    public void handleErrorJob(Job job) {
        model.addFirst(job);
    }

    public void pollBatch(String threadName, WorkerState state)
            throws InterruptedException, ExecutionException {
        model.pollBatch(
                threadName, (state == WorkerState.WORKER_MODEL_LOADED) ? 0 : Long.MAX_VALUE, jobs);
    }

    public void shutdown() {
        return;
    }

    public void startEventDispatcher() {
        return;
    }
}
