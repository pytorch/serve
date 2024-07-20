package org.pytorch.serve.workflow.messages;

import com.google.gson.annotations.SerializedName;
import io.netty.handler.codec.http.QueryStringDecoder;
import org.pytorch.serve.util.NettyUtils;

public class RegisterWorkflowRequest {
    @SerializedName("workflow_name")
    private String workflowName;

    @SerializedName("response_timeout")
    private int responseTimeout;

    @SerializedName("startup_timeout")
    private int startupTimeout;

    @SerializedName("url")
    private String workflowUrl;

    @SerializedName("s3_sse_kms")
    private boolean s3SseKms;

    public RegisterWorkflowRequest(QueryStringDecoder decoder) {
        workflowName = NettyUtils.getParameter(decoder, "workflow_name", null);
        responseTimeout = NettyUtils.getIntParameter(decoder, "response_timeout", 120);
        startupTimeout = NettyUtils.getIntParameter(decoder, "startup_timeout", 120);
        workflowUrl = NettyUtils.getParameter(decoder, "url", null);
        s3SseKms = Boolean.parseBoolean(NettyUtils.getParameter(decoder, "s3_sse_kms", "false"));
    }

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

    public int getStartupTimeout() {
        return startupTimeout;
    }

    public void setStartupTimeout(int startupTimeout) {
        this.startupTimeout = startupTimeout;
    }

    public String getWorkflowUrl() {
        return workflowUrl;
    }

    public void setWorkflowUrl(String workflowUrl) {
        this.workflowUrl = workflowUrl;
    }

    public Boolean getS3SseKms() {
        return s3SseKms;
    }
}
