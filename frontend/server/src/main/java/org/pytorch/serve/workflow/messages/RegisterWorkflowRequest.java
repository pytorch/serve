package org.pytorch.serve.workflow.messages;

import com.google.gson.annotations.SerializedName;
import io.netty.handler.codec.http.QueryStringDecoder;

public class RegisterWorkflowRequest {
    @SerializedName("workflow_name")
    private String workflowName;

    @SerializedName("response_timeout")
    private int responseTimeout;

    @SerializedName("url")
    private String workflowUrl;


    public RegisterWorkflowRequest(QueryStringDecoder decoder) {}

    public void setWorkflowName(String workflowName) {
        this.workflowName = workflowName;
    }

    public String getWorkflowName() {
        return workflowName;
    }

    public int getResponseTimeout() {
        return responseTimeout;
    }

    public void setResponseTimeout(int responseTimeout) {
        this.responseTimeout = responseTimeout;
    }

    public String getWorkflowUrl() {
        return workflowUrl;
    }

    public void setWorkflowUrl(String workflowUrl) {
        this.workflowUrl = workflowUrl;
    }

}
