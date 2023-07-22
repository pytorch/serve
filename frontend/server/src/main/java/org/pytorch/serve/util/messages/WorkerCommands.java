package org.pytorch.serve.util.messages;

import com.google.gson.annotations.SerializedName;

public enum WorkerCommands {
    @SerializedName("predict")
    PREDICT("predict"),
    @SerializedName("load")
    LOAD("load"),
    @SerializedName("unload")
    UNLOAD("unload"),
    @SerializedName("stats")
    STATS("stats"),
    @SerializedName("describe")
    DESCRIBE("describe"),
    @SerializedName("streampredict")
    STREAMPREDICT("streampredict");

    private String command;

    WorkerCommands(String command) {
        this.command = command;
    }

    public String getCommand() {
        return command;
    }
}
