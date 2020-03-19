package org.pytorch.serve.servingsdk.impl;

import org.pytorch.serve.wlm.WorkerState;
import org.pytorch.serve.wlm.WorkerThread;
import software.amazon.ai.mms.servingsdk.Worker;

public class ModelWorker implements Worker {
    private boolean running;
    private long memory;

    public ModelWorker(WorkerThread t) {
        running = t.getState() == WorkerState.WORKER_MODEL_LOADED;
        memory = t.getMemory();
    }

    @Override
    public boolean isRunning() {
        return running;
    }

    @Override
    public long getWorkerMemory() {
        return memory;
    }
}
