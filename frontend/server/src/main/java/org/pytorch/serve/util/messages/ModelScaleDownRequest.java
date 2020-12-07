package org.pytorch.serve.util.messages;

import org.pytorch.serve.wlm.Model;

public class ModelScaleDownRequest extends BaseModelRequest {

    private String port;

    public ModelScaleDownRequest(Model model, String port) {
        super(WorkerCommands.SCALE_DOWN, model.getModelName());
        this.port = port;
    }

    public String getPort() {
        return this.port;
    }
}
