package org.pytorch.serve.servingsdk.metrics;

public class DimensionRegistry {
    public static final String LEVEL = "Level";
    public static final String MODELNAME = "ModelName";
    public static final String MODELVERSION = "ModelVersion";
    public static final String WORKERNAME = "WorkerName";

    public static class LevelRegistry {
        public static final String MODEL = "Model";
        public static final String HOST = "Host";
        public static final String WORKER = "Worker";
    }
}
