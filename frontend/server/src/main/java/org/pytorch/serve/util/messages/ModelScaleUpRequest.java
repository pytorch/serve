package org.pytorch.serve.util.messages;

import org.pytorch.serve.wlm.Model;

public class ModelScaleUpRequest extends BaseModelRequest {

    private String sockType;
    private String sockName;
    private String host;
    private String port;
    private String fifoPath;


    public ModelScaleUpRequest(Model model, String sockType, String sockName, String host, String port, String fifoPath) {
        super(WorkerCommands.SCALE_UP, model.getModelName());
        this.port = port;
        this.sockType = sockType;
        this.fifoPath = fifoPath;
        this.host = host;
        this.sockName = sockName;
    }

    public String getSockType() {
        return this.sockType;
    }

    public String getSockName() {
        return this.sockName;
    }
    public String getHost() {
        return this.host;
    }

    public String getPort() {
        return this.port;
    }

    public String getFifoPath() {
        return this.fifoPath;
    }

}
