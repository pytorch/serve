package org.pytorch.serve.wlm;

import io.netty.handler.codec.http.HttpResponseStatus;
import java.util.LinkedHashMap;
import java.util.Map;
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

    private Model model;
    private Map<String, Job> jobs;

    public BatchAggregator(Model model) {
        this.model = model;
        jobs = new LinkedHashMap<>();
    }

    public BaseModelRequest getRequest(String threadName, WorkerState state)
            throws InterruptedException {
        jobs.clear();

        ModelInferenceRequest req = new ModelInferenceRequest(model.getModelName());

        model.pollBatch(
                threadName, (state == WorkerState.WORKER_MODEL_LOADED) ? 0 : Long.MAX_VALUE, jobs);

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
                j.setScheduled();
                req.addRequest(j.getPayload());
            }
        }
        return req;
    }

    public void sendResponse(ModelWorkerResponse message) {
        // TODO: Handle prediction level code

        if (message.getCode() == 200) {
            if (jobs.isEmpty()) {
                // this is from initial load.
                return;
            }

            for (Predictions prediction : message.getPredictions()) {
                String jobId = prediction.getRequestId();
                Job job = jobs.remove(jobId);
                if (job == null) {
                    throw new IllegalStateException("Unexpected job: " + jobId);
                }
                job.response(
                        prediction.getResp(),
                        prediction.getContentType(),
                        prediction.getStatusCode(),
                        prediction.getReasonPhrase(),
                        prediction.getHeaders());
            }
        } else {
            for (String reqId : jobs.keySet()) {
                Job j = jobs.remove(reqId);
                if (j == null) {
                    throw new IllegalStateException("Unexpected job: " + reqId);
                }
                j.sendError(HttpResponseStatus.valueOf(message.getCode()), message.getMessage());
            }
            if (!jobs.isEmpty()) {
                throw new IllegalStateException("Not all jobs get response.");
            }
        }
    }

    public void sendError(BaseModelRequest message, String error, HttpResponseStatus status) {
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
                    logger.error("Unexpected job: " + requestId);
                } else {
                    job.sendError(status, error);
                }
            }
            if (!jobs.isEmpty()) {
                jobs.clear();
                logger.error("Not all jobs get response.");
            }
        } else {
            // Send the error message to all the jobs
            for (Map.Entry<String, Job> j : jobs.entrySet()) {
                String jobsId = j.getValue().getJobId();
                Job job = jobs.remove(jobsId);

                if (job.isControlCmd()) {
                    job.sendError(status, error);
                } else {
                    // Data message can be handled by other workers.
                    // If batch has gone past its batch max delay timer?
                    model.addFirst(job);
                }
            }
        }
    }
}
