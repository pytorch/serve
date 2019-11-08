package org.pytorch.serve.openapi;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Operation {

    private String summary;
    private String description;
    private String operationId;
    private List<Parameter> parameters = new ArrayList<>();
    private RequestBody requestBody;
    private Map<String, Response> responses;
    private Boolean deprecated;

    public Operation() {}

    public Operation(String operationId) {
        this(operationId, null);
    }

    public Operation(String operationId, String description) {
        this.operationId = operationId;
        this.description = description;
    }

    public String getSummary() {
        return summary;
    }

    public void setSummary(String summary) {
        this.summary = summary;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getOperationId() {
        return operationId;
    }

    public void setOperationId(String operationId) {
        this.operationId = operationId;
    }

    public List<Parameter> getParameters() {
        return parameters;
    }

    public void setParameters(List<Parameter> parameters) {
        this.parameters = parameters;
    }

    public void addParameter(Parameter parameter) {
        if (parameters == null) {
            parameters = new ArrayList<>();
        }
        parameters.add(parameter);
    }

    public RequestBody getRequestBody() {
        return requestBody;
    }

    public void setRequestBody(RequestBody requestBody) {
        this.requestBody = requestBody;
    }

    public Map<String, Response> getResponses() {
        return responses;
    }

    public void setResponses(Map<String, Response> responses) {
        this.responses = responses;
    }

    public void addResponse(Response response) {
        if (responses == null) {
            responses = new LinkedHashMap<>();
        }
        responses.put(response.getCode(), response);
    }

    public Boolean getDeprecated() {
        return deprecated;
    }

    public void setDeprecated(Boolean deprecated) {
        this.deprecated = deprecated;
    }
}
