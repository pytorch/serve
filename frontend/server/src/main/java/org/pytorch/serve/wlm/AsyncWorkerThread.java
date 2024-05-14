package org.pytorch.serve.wlm;

import static org.pytorch.serve.wlm.WorkerThread.logger;
import static org.pytorch.serve.wlm.WorkerThread.loggerTelemetryMetrics;

import java.net.HttpURLConnection;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.WorkerCommands;

import io.netty.channel.EventLoopGroup;

public class AsyncWorkerThread extends WorkerThread{
    public AsyncWorkerThread(
            ConfigManager configManager,
            EventLoopGroup backendEventGroup,
            int port,
            int gpuId,
            Model model,
            BatchAggregator aggregator,
            WorkerStateListener listener) {
        super(configManager, backendEventGroup, port, gpuId, model, aggregator, listener);
        assert 0 == 1;
    }

    @Override
    public void run() {
        responseTimeout = model.getResponseTimeout();
        Thread thread = Thread.currentThread();
        thread.setName(getWorkerName());
        currentThread.set(thread);
        req = null;
        int status = HttpURLConnection.HTTP_INTERNAL_ERROR;

        try {
            connect();

            while (isRunning()) {
                req = aggregator.getRequest(workerId, state);
                WorkerCommands workerCmd = req.getCommand();

                long wtStartTime = System.currentTimeMillis();
                int repeats = getRepeats(workerCmd);
                logger.debug(
                        "Flushing req.cmd {} repeats {} to backend at: {}",
                        workerCmd,
                        repeats,
                        wtStartTime);
                List<CompletableFuture<Void>> futureRequests = new ArrayList<>(repeats);
                for (int i = 0; backendChannel.size() > 0 && i < repeats; i++) {
                    int idx = i;
                    futureRequests.add(
                            CompletableFuture.runAsync(
                                    () -> {
                                        try {
                                            backendChannel.get(idx).writeAndFlush(req).sync();
                                        } catch (InterruptedException e) {
                                            logger.error("Failed to send request to backend", e);
                                        }
                                    }));
                }

                futureRequests.stream().map(CompletableFuture::join);

                ModelWorkerResponse reply = null;

                boolean jobDone = false;
                long totalDuration = 0;

                logger.info("Looping backend response at: {}", System.currentTimeMillis());

                do {
                    long begin = System.currentTimeMillis();
                    for (int i = 0; i < repeats; i++) {
                        reply = replies.poll(responseTimeout, TimeUnit.SECONDS);
                        if (req.getCommand() != WorkerCommands.LOAD) {
                            break;
                        }
                    }

                    long duration = System.currentTimeMillis() - begin;

                    if (reply != null) {
                        jobDone = aggregator.sendResponse(reply);
                    } else if (req.getCommand() != WorkerCommands.DESCRIBE) {
                        int val = model.incrFailedInfReqs();
                        logger.error("Number or consecutive unsuccessful inference {}", val);
                        throw new WorkerInitializationException(
                                "Backend worker did not respond in given time");
                    }
                    totalDuration += duration;
                } while (!jobDone);
                logger.info("Backend response time: {}", totalDuration);

                switch (req.getCommand()) {
                    case PREDICT:
                        model.resetFailedInfReqs();
                        break;
                    case STREAMPREDICT:
                        model.resetFailedInfReqs();
                        break;
                    case LOAD:
                        if (reply.getCode() == 200) {
                            setState(WorkerState.WORKER_MODEL_LOADED, HttpURLConnection.HTTP_OK);
                            backoffIdx = 0;
                        } else {
                            setState(WorkerState.WORKER_ERROR, reply.getCode());
                            status = reply.getCode();
                        }
                        break;
                    case DESCRIBE:
                        if (reply == null) {
                            aggregator.sendError(
                                    req,
                                    "Failed to get customized model matadata.",
                                    HttpURLConnection.HTTP_INTERNAL_ERROR);
                        }
                        break;
                    case UNLOAD:
                    case STATS:
                    default:
                        break;
                }
                req = null;
                double workerThreadTime =
                        (System.currentTimeMillis() - wtStartTime) - totalDuration;
                if (this.workerThreadTimeMetric != null) {
                    try {
                        this.workerThreadTimeMetric.addOrUpdate(
                                this.workerThreadTimeMetricDimensionValues, workerThreadTime);
                    } catch (Exception e) {
                        logger.error("Failed to update frontend metric WorkerThreadTime: ", e);
                    }
                }
            }
        } catch (InterruptedException e) {
            logger.debug("System state is : " + state);
            if (state == WorkerState.WORKER_SCALED_DOWN || state == WorkerState.WORKER_STOPPED) {
                logger.debug("Shutting down the thread .. Scaling down.");
            } else {
                logger.debug(
                        "Backend worker monitoring thread interrupted or backend worker process died., responseTimeout:"
                                + responseTimeout
                                + "sec",
                        e);
            }
        } catch (WorkerInitializationException e) {
            logger.error("Backend worker error", e);
        } catch (OutOfMemoryError oom) {
            logger.error("Out of memory error when creating workers", oom);
            status = HttpURLConnection.HTTP_ENTITY_TOO_LARGE;
            if (ConfigManager.getInstance().isTelemetryEnabled()) {
                loggerTelemetryMetrics.info(
                        "ModelServerError.Count:1|#TorchServe:{},{}:-1",
                        ConfigManager.getInstance().getVersion(),
                        oom.getClass().getCanonicalName());
            }
        } catch (IllegalStateException e) {
            logger.error("IllegalStateException error", e);
        } catch (Throwable t) {
            logger.warn("Backend worker thread exception.", t);
            if (ConfigManager.getInstance().isTelemetryEnabled()) {
                loggerTelemetryMetrics.info(
                        "ModelServerError.Count:1|#TorchServe:{},{}:-1",
                        ConfigManager.getInstance().getVersion(),
                        t.getClass().getCanonicalName());
            }
        } finally {
            // WorkerThread is running in thread pool, the thread will be assigned to next
            // Runnable once this worker is finished. If currentThread keep holding the reference
            // of the thread, currentThread.interrupt() might kill next worker.
            for (int i = 0;
                    backendChannel.size() > 0
                            && i < (model.getParallelLevel() > 0 ? model.getParallelLevel() : 1);
                    i++) {
                backendChannel.get(i).disconnect();
            }
            backendChannel.clear();
            currentThread.set(null);
            Integer exitValue = lifeCycle.getExitValue();

            if (exitValue != null && exitValue == 137) {
                status = HttpURLConnection.HTTP_ENTITY_TOO_LARGE;
            }

            if (req != null) {
                aggregator.sendError(req, "Worker died.", status);
            }
            aggregator.cleanJobs();
            setState(WorkerState.WORKER_STOPPED, status);
            lifeCycle.exit();
            if (isHealthy()) { // still within maxRetryTimeoutInMill window
                retry();
            }
        }
    }

}
