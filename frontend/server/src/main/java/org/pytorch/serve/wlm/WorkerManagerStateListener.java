package org.pytorch.serve.wlm;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

public class WorkerManagerStateListener {

    // TODO : Can you refactor states to remove this possible redundant code

    private CompletableFuture<Integer> future;
    private AtomicInteger count;

    public WorkerManagerStateListener(CompletableFuture<Integer> future, int count) {
        this.future = future;
        this.count = new AtomicInteger(count);
    }

    public void notifyChangeState(String modelName, WorkerManagerState state, int status) {
        // Update success and fail counts
        if (state == WorkerManagerState.WORKER_MODEL_LOADED) {
            if (count.decrementAndGet() == 0) {
                future.complete(status);
            }
        }
        if (state == WorkerManagerState.WORKER_ERROR
                || state == WorkerManagerState.WORKER_STOPPED) {
            future.complete(status);
        }
    }
}
