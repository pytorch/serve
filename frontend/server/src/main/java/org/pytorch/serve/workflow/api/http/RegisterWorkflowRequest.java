package org.pytorch.serve.workflow.api.http;

import io.netty.handler.codec.http.QueryStringDecoder;

public class RegisterWorkflowRequest {
    private String wfName;

    private String responseTimeout;

    public RegisterWorkflowRequest(QueryStringDecoder decoder) {}

    public void setWfName(String wfName) {
        this.wfName = wfName;
    }

    public String getWfName() {
        return wfName;
    }

    public String getResponseTimeout() {
        return responseTimeout;
    }

    public void setResponseTimeout(String responseTimeout) {
        this.responseTimeout = responseTimeout;
    }
}
