package org.pytorch.serve.util.messages;

import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;

public class ModelLoadModelRequest extends BaseModelRequest {

    /**
     * ModelLoadModelRequest is a interface between frontend and backend to notify the backend to
     * load a particular model.
     */
    private String modelPath;

    private String handler;
    private String envelope;
    private int batchSize;
    private int gpuId;
    private boolean limitMaxImagePixels;

    public ModelLoadModelRequest(Model model, int gpuId) {
        super(WorkerCommands.LOAD, model.getModelName());
        this.gpuId = gpuId;
        modelPath = model.getModelDir().getAbsolutePath();
        handler = model.getModelArchive().getManifest().getModel().getHandler();
        envelope = model.getModelArchive().getManifest().getModel().getEnvelope();
        batchSize = model.getBatchSize();
        limitMaxImagePixels = ConfigManager.getInstance().isLimitMaxImagePixels();
    }

    public String getModelPath() {
        return modelPath;
    }

    public String getHandler() {
        return handler;
    }

    public String getEnvelope() {
        return envelope;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getGpuId() {
        return gpuId;
    }

    public boolean isLimitMaxImagePixels() {
        return limitMaxImagePixels;
    }
}
