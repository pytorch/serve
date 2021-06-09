package org.pytorch.serve.wlm;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

public class WorkerStateListener {

    private CompletableFuture<Integer> future;
    private AtomicInteger count;

    public WorkerStateListener(CompletableFuture<Integer> future, int count) {
        this.future = future;
        this.count = new AtomicInteger(count);
    }

    public void notifyChangeState(String modelName, WorkerState state, Integer status) {
        // Update success and fail counts
        if (state == WorkerState.WORKER_MODEL_LOADED) {
            if (count.decrementAndGet() == 0) {
                future.complete(status);
            }
        }
        if (state == WorkerState.WORKER_ERROR || state == WorkerState.WORKER_STOPPED) {
            future.complete(status);
        }
    }
}
