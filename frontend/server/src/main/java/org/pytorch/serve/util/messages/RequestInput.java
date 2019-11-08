package org.pytorch.serve.util.messages;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RequestInput {

    private String requestId;
    private Map<String, String> headers;
    private List<InputParameter> parameters;

    public RequestInput(String requestId) {
        this.requestId = requestId;
        headers = new HashMap<>();
        parameters = new ArrayList<>();
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public Map<String, String> getHeaders() {
        return headers;
    }

    public void setHeaders(Map<String, String> headers) {
        this.headers = headers;
    }

    public void updateHeaders(String key, String val) {
        headers.put(key, val);
    }

    public List<InputParameter> getParameters() {
        return parameters;
    }

    public void setParameters(List<InputParameter> parameters) {
        this.parameters = parameters;
    }

    public void addParameter(InputParameter modelInput) {
        parameters.add(modelInput);
    }

    public String getStringParameter(String key) {
        for (InputParameter param : parameters) {
            if (key.equals(param.getName())) {
                return new String(param.getValue(), StandardCharsets.UTF_8);
            }
        }
        return null;
    }
}
