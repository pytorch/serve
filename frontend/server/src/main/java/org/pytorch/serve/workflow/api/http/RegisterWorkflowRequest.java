package org.pytorch.serve.workflow.api.http;

import io.netty.handler.codec.http.QueryStringDecoder;

public class RegisterWorkflowRequest {
    private String wfName;

    public RegisterWorkflowRequest(QueryStringDecoder decoder) {}

    public void setWfName(String wfName) {
        this.wfName = wfName;
    }

    public String getWfName() {
        return wfName;
    }
}
