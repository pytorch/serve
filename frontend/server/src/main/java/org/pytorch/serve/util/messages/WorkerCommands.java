package org.pytorch.serve.util.messages;

import com.google.gson.annotations.SerializedName;

public enum WorkerCommands {
    @SerializedName("predict")
    PREDICT("predict"),
    @SerializedName("load")
    LOAD("load"),
    @SerializedName("unload")
    UNLOAD("unload"),
    @SerializedName("scale_up")
    SCALE_UP("scale_up"),
    @SerializedName("scale_down")
    SCALE_DOWN("scale_down"),
    @SerializedName("stats")
    STATS("stats");

    private String command;

    WorkerCommands(String command) {
        this.command = command;
    }

    public String getCommand() {
        return command;
    }
}
