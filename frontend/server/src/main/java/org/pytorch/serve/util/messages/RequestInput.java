package org.pytorch.serve.util.messages;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RequestInput {
    public static final String TS_STREAM_NEXT = "ts_stream_next";

    private String requestId;
    private Map<String, String> headers;
    private List<InputParameter> parameters;
    private long clientExpireTS;

    public RequestInput(String requestId) {
        this.requestId = requestId;
        headers = new HashMap<>();
        parameters = new ArrayList<>();
        clientExpireTS = Long.MAX_VALUE; // default(never expire): Long.MAX_VALUE
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

    public long getClientExpireTS() {
        return clientExpireTS;
    }

    public void setClientExpireTS(long clientTimeoutInMills) {
        if (clientTimeoutInMills > 0) {
            this.clientExpireTS = System.currentTimeMillis() + clientTimeoutInMills;
        }
    }
}
