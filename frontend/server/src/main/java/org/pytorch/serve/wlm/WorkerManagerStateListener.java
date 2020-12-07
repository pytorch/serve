package org.pytorch.serve.wlm;

import io.netty.handler.codec.http.HttpResponseStatus;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

public class WorkerManagerStateListener {

    // TODO : Can you refactor states to remove this possible redundant code

    private CompletableFuture<HttpResponseStatus> future;
    private AtomicInteger count;

    public WorkerManagerStateListener(CompletableFuture<HttpResponseStatus> future, int count) {
        this.future = future;
        this.count = new AtomicInteger(count);
    }

    public void notifyChangeState(String modelName, WorkerManagerState state, HttpResponseStatus status) {
        // Update success and fail counts
        if (state == WorkerManagerState.WORKER_MODEL_LOADED) {
            if (count.decrementAndGet() == 0) {
                future.complete(status);
            }
        }
        if (state == WorkerManagerState.WORKER_ERROR || state == WorkerManagerState.WORKER_STOPPED) {
            future.complete(status);
        }
    }
}
