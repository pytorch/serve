package org.pytorch.serve.util.messages;

import org.pytorch.serve.wlm.Model;

public class ModelLoadModelRequest extends BaseModelRequest {

    /**
     * ModelLoadModelRequest is a interface between frontend and backend to notify the backend to
     * load a particular model.
     */
    private String modelPath;

    private String handler;
    private int batchSize;
    private int gpuId;
    private String ioFileDescriptor;

    public ModelLoadModelRequest(Model model, int gpuId, String fd) {
        super(WorkerCommands.LOAD, model.getModelName());
        this.gpuId = gpuId;
        modelPath = model.getModelDir().getAbsolutePath();
        handler = model.getModelArchive().getManifest().getModel().getHandler();
        batchSize = model.getBatchSize();
        ioFileDescriptor = fd;
    }

    public String getIoFileDescriptor() {
        return ioFileDescriptor;
    }

    public void setIoFileDescriptor(String ioFileDescriptor) {
        this.ioFileDescriptor = ioFileDescriptor;
    }

    public String getModelPath() {
        return modelPath;
    }

    public String getHandler() {
        return handler;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getGpuId() {
        return gpuId;
    }
}
