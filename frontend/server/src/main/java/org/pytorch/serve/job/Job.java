package org.pytorch.serve.job;

import java.util.Map;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;

public abstract class Job {

    private String modelName;
    private String modelVersion;
    private WorkerCommands cmd; // Else its data msg or inf requests
    private RequestInput input;
    private long begin;
    private long scheduled;

    public Job(String modelName, String version, WorkerCommands cmd, RequestInput input) {
        this.modelName = modelName;
        this.cmd = cmd;
        this.input = input;
        this.modelVersion = version;
        begin = System.nanoTime();
        scheduled = begin;
    }

    public String getJobId() {
        return input.getRequestId();
    }

    public String getModelName() {
        return modelName;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public WorkerCommands getCmd() {
        return cmd;
    }

    public boolean isControlCmd() {
        switch (cmd) {
            case PREDICT:
            case OIPPREDICT:
            case STREAMPREDICT:
            case STREAMPREDICT2:
            case DESCRIBE:
                return false;
            default:
                return true;
        }
    }

    public RequestInput getPayload() {
        return input;
    }

    public void setScheduled() {
        scheduled = System.nanoTime();
    }

    public long getBegin() {
        return begin;
    }

    public long getScheduled() {
        return scheduled;
    }

    public abstract void response(
            byte[] body,
            CharSequence contentType,
            int statusCode,
            String statusPhrase,
            Map<String, String> responseHeaders);

    public abstract void sendError(int status, String error);

    public String getGroupId() {
        if (input != null) {
            return input.getSequenceId();
        }
        return null;
    }

    public abstract boolean isOpen();
}
