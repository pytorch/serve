package org.pytorch.serve.util.messages;

public class BaseModelRequest {

    private WorkerCommands command;
    private String modelName;

    public BaseModelRequest() {}

    public BaseModelRequest(WorkerCommands command, String modelName) {
        this.command = command;
        this.modelName = modelName;
    }

    public WorkerCommands getCommand() {
        return command;
    }

    public void setCommand(WorkerCommands workerCommands) {
        this.command = workerCommands;
    }

    public String getModelName() {
        return modelName;
    }
}
