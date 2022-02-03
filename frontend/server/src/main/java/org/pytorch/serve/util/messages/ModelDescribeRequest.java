package org.pytorch.serve.util.messages;

import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;

public class ModelDescribeRequest extends BaseModelRequest {

    /**
     * ModelDescribeRequest is a interface between frontend and backend to fetch customized model
     * metadata from backend.
     */

    public ModelDescribeRequest(String modelName) {
        super(WorkerCommands.DESCRIBE, modelName);
    }
}
