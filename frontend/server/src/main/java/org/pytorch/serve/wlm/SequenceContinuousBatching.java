package org.pytorch.serve.wlm;

import java.util.Map;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.JobGroup;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.ModelInferenceRequest;
import org.pytorch.serve.util.messages.ModelLoadModelRequest;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.Predictions;
import org.pytorch.serve.util.messages.RequestInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SequenceContinuousBatching extends SequenceBatching {
    private static final Logger logger = LoggerFactory.getLogger(SequenceContinuousBatching.class);

    public SequenceContinuousBatching(Model model) {
        super(model);
    }

    @Override
    public BaseModelRequest getRequest(String threadName, WorkerState state)
            throws InterruptedException, ExecutionException {

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
     *     stream response (not include the last stream) is sent This is a copy of sendResponse from
     *     ContinuousBatching + 1. setJobGroupFinished: handle a list of jobGroups end. 2.
     *     resetCurrentJobGroupIds
     */
    @Override
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
                if (streamNext == null || (streamNext != null && streamNext.equals("false"))) {
                    jobs.remove(jobId);
                } else if (!job.isOpen()) {
                    jobs.remove(job.getJobId());
                    logger.info(
                            "Connection to client got closed; Removing job: {}",
                            job.getPayload().getRequestId());
                } else {
                    job.getPayload().setCachedInBackend(true);
                }
                setJobGroupFinished(prediction);
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

        resetCurrentJobGroupIds();

        return true;
    }

    private void setJobGroupFinished(Predictions prediction) {
        String val =
                prediction
                        .getHeaders()
                        .getOrDefault(
                                ConfigManager.getInstance().getTsHeaderKeySequenceEnd(), null);
        if (val != null) {
            String[] jobGroupIds = val.split(";");
            for (String j : jobGroupIds) {
                String jobGroupId = j.trim();
                JobGroup jobGroup = model.getJobGroup(jobGroupId);
                if (jobGroup != null) {
                    jobGroup.setFinished(true);
                }
            }
        }
    }

    @Override
    protected void pollInferJob() throws InterruptedException {
        // TBD: Temporarily hard code the continuous batch size is 2 * batchSize
        model.pollInferJob(jobs, model.getBatchSize() * 2 - jobs.size(), jobsQueue);

        for (Job job : jobs.values()) {
            if (job.getGroupId() != null) {
                currentJobGroupIds.add(job.getGroupId());
            }
        }
    }

    private void resetCurrentJobGroupIds() {
        if (!currentJobGroupIds.isEmpty()) {
            eventJobGroupIds.addAll(currentJobGroupIds);
            currentJobGroupIds.clear();
        }
        return;
    }
}
