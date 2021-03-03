package org.pytorch.serve.wlm;

public enum WorkerManagerState {
    WORKER_STARTED,
    WORKER_MODEL_LOADED,
    WORKER_STOPPED,
    WORKER_ERROR,
    WORKER_SCALED_DOWN
}
