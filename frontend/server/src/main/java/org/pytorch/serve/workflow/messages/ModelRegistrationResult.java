package org.pytorch.serve.workflow.messages;

import org.pytorch.serve.http.StatusResponse;

public class ModelRegistrationResult {
    private final String modelName;
    private final StatusResponse response;

    public ModelRegistrationResult(String modelName, StatusResponse response) {
        this.modelName = modelName;
        this.response = response;
    }

    public String getModelName() {
        return modelName;
    }

    public StatusResponse getResponse() {
        return response;
    }
}
