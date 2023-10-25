package org.pytorch.serve.util.messages;

import java.util.ArrayList;
import java.util.List;

public class ModelInferenceRequest extends BaseModelRequest {

    private List<RequestInput> batch;

    public ModelInferenceRequest(String modelName) {
        super(WorkerCommands.PREDICT, modelName);
        batch = new ArrayList<>();
    }

    public List<RequestInput> getRequestBatch() {
        return batch;
    }

    public void setRequestBatch(List<RequestInput> requestBatch) {
        this.batch = requestBatch;
    }

    public void addRequest(RequestInput req) {
        batch.add(req);
    }

    public void setCachedInBackend(boolean cached) {
        for (RequestInput input : batch) {
            input.setCachedInBackend(cached);
        }
    }
}
