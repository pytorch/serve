package org.pytorch.serve.wlm;

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

public class ContinuousBatching extends BatchAggregator {
    private static final Logger logger = LoggerFactory.getLogger(ContinuousBatching.class);

    public ContinuousBatching(Model model) {
        super(model);
    }

    public BaseModelRequest getRequest(String threadName, WorkerState state)
            throws InterruptedException, ExecutionException {
        int batchQuota = model.getBatchSize() - jobs.size();

        ModelInferenceRequest req = new ModelInferenceRequest(model.getModelName());

        pollBatch(threadName, state, batchQuota);

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
        // TODO: Handle prediction level code
        if (message.getCode() == 200) {
            if (message.getPredictions().isEmpty()) {
                // The jobs size is always 1 in the case control command
                for (Map.Entry<String, Job> j : jobs.entrySet()) {
                    Job job = j.getValue();
                    if (job.isControlCmd()) {
                        cleanJobs();
                        return true;
                    }
                }
            }
            for (Predictions prediction : message.getPredictions()) {
                String jobId = prediction.getRequestId();
                Job job = jobs.get(jobId);

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
                if (streamNext != null && streamNext.equals("false")) {
                    jobs.remove(jobId);
                } else if (!job.isOpen()) {
                    jobs.remove(job.getJobId());
                    logger.info(
                            "Connection to client got closed; Removing job: {}",
                            job.getPayload().getRequestId());
                } else {
                    job.getPayload().setCachedInBackend(true);
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
            cleanJobs();
        }

        return true;
    }

    private void pollBatch(String threadName, WorkerState state, int batchSize)
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
            model.pollInferJob(jobs, batchSize);
        }
    }
}
