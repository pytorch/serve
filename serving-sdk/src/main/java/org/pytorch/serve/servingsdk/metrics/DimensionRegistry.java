package org.pytorch.serve.servingsdk.metrics;

/**
 * This is a registry for different metric dimensions and their values.
 * The purpose of this registry is to list and refer all the Dimensions available in TorchServe in one place.
 * It is expected whenever a new metric Dimension get added in TorchServe, a corresponding entry should be made here.
 * Plugin writers then will be able to refer this list.
 * This is not exhaustive list. Dimension can be added to Metric without adding it to this Registry.
 */
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
