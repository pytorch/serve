package org.pytorch.serve.wlm;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.ModelInferenceRequest;
import org.pytorch.serve.util.messages.ModelLoadModelRequest;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.Predictions;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AsyncBatchAggregator extends BatchAggregator {
    protected Map<String, Job> jobs_in_backend;

    private static final Logger logger = LoggerFactory.getLogger(AsyncBatchAggregator.class);

    public AsyncBatchAggregator() {
        super();
    }

    public AsyncBatchAggregator(Model model) {
        super(model);
        jobs_in_backend = new LinkedHashMap<>();
    }

    @Override
    public BaseModelRequest getRequest(String threadName, WorkerState state)
            throws InterruptedException, ExecutionException {

        logger.info("Getting requests from model: {}", model);
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
                if (j.getCmd() == WorkerCommands.STREAMPREDICT
                        || j.getCmd() == WorkerCommands.STREAMPREDICT2) {
                    req.setCommand(j.getCmd());
                }
                req.addRequest(j.getPayload());
                jobs_in_backend.put(j.getJobId(), j);
                jobs.remove(j.getJobId());
            }
        }
        return req;
    }

    /**
     * @param message: a response of a batch inference requests
     * @return - true: either a non-stream response or last stream response is sent - false: a
     *     stream response (not include the last stream) is sent
     */
    @Override
    public boolean sendResponse(ModelWorkerResponse message) {
        boolean jobDone = true;
        // TODO: Handle prediction level code
        if (message.getCode() == 200) {
            if (message.getPredictions().isEmpty()) {
                // this is from initial load.
                logger.info("Predictions is empty. This is from initial load....");
                jobs.clear();
                // jobs_in_backend.clear();
                return true;
            }
            for (Predictions prediction : message.getPredictions()) {
                String jobId = prediction.getRequestId();
                Job job = jobs_in_backend.get(jobId);

                if (job == null) {
                    throw new IllegalStateException(
                            "Unexpected job in sendResponse() with 200 status code: " + jobId);
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
                String streamNext =
                        prediction
                                .getHeaders()
                                .get(org.pytorch.serve.util.messages.RequestInput.TS_STREAM_NEXT);
                if ("false".equals(streamNext)) {
                    jobs_in_backend.remove(jobId);
                }
            }

        } else {
            for (Map.Entry<String, Job> j : jobs_in_backend.entrySet()) {
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
        return false;
    }

    @Override
    public void sendError(BaseModelRequest message, String error, int status) {
        if (message instanceof ModelLoadModelRequest) {
            logger.warn("Load model failed: {}, error: {}", message.getModelName(), error);
            return;
        }

        if (message != null) {
            ModelInferenceRequest msg = (ModelInferenceRequest) message;
            for (RequestInput req : msg.getRequestBatch()) {
                String requestId = req.getRequestId();
                Job job = jobs_in_backend.remove(requestId);
                if (job == null) {
                    logger.error("Unexpected job in sendError(): " + requestId);
                } else {
                    job.sendError(status, error);
                }
            }
            if (!jobs_in_backend.isEmpty()) {
                // cleanJobs();
                logger.error("Not all jobs got an error response.");
            }
        } else {
            // Send the error message to all the jobs
            List<Map.Entry<String, Job>> entries = new ArrayList<>(jobs_in_backend.entrySet());
            for (Map.Entry<String, Job> j : entries) {
                String jobsId = j.getValue().getJobId();
                Job job = jobs_in_backend.remove(jobsId);

                if (job.isControlCmd()) {
                    job.sendError(status, error);
                } else {
                    // Data message can be handled by other workers.
                    // If batch has gone past its batch max delay timer?
                    handleErrorJob(job);
                }
            }
        }
    }

    @Override
    public void handleErrorJob(Job job) {
        model.addFirst(job);
    }

    @Override
    public void pollBatch(String threadName, WorkerState state)
            throws InterruptedException, ExecutionException {
        Map<String, Job> newJobs = new LinkedHashMap<>();
        model.pollBatch(
                threadName,
                (state == WorkerState.WORKER_MODEL_LOADED) ? 0 : Long.MAX_VALUE,
                newJobs);
        for (Job job : newJobs.values()) {
            jobs.put(job.getJobId(), job);
            logger.debug("Adding job to jobs: {}", job.getJobId());
        }
    }
}
