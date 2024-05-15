package org.pytorch.serve.wlm;

import static org.pytorch.serve.wlm.WorkerThread.logger;
import static org.pytorch.serve.wlm.WorkerThread.loggerTelemetryMetrics;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.SocketAddress;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ConcurrentHashMap;

import org.pytorch.serve.grpc.inference.PredictionResponse;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.pytorch.serve.util.codec.ModelResponseDecoder;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.Predictions;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.pytorch.serve.util.messages.ModelInferenceRequest;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;

public class AsyncWorkerThread extends WorkerThread{
    // protected ConcurrentHashMap requestsInBackend;
    protected boolean loadingFinished;

    public AsyncWorkerThread(
            ConfigManager configManager,
            EventLoopGroup backendEventGroup,
            int port,
            int gpuId,
            Model model,
            BatchAggregator aggregator,
            WorkerStateListener listener) {
        super(configManager, backendEventGroup, port, gpuId, model, aggregator, listener);
        loadingFinished = false;
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
                // for (int i = 0; backendChannel.size() > 0 && i < repeats; i++) {
                int idx = 0;
                futureRequests.add(
                        CompletableFuture.runAsync(
                                () -> {
                                    try {
                                        backendChannel.get(idx).writeAndFlush(req).sync();
                                    } catch (InterruptedException e) {
                                        logger.error("Failed to send request to backend", e);
                                    }
                                }));
                // }

                futureRequests.stream().map(CompletableFuture::join);

                // ModelWorkerResponse reply = null;

                // boolean jobDone = false;
                // long totalDuration = 0;

                // logger.info("Looping backend response at: {}", System.currentTimeMillis());

                // do {
                //     long begin = System.currentTimeMillis();
                //     for (int i = 0; i < repeats; i++) {
                //         reply = replies.poll(responseTimeout, TimeUnit.SECONDS);
                //         if (req.getCommand() != WorkerCommands.LOAD) {
                //             break;
                //         }
                //     }

                //     long duration = System.currentTimeMillis() - begin;

                //     if (reply != null) {
                //         jobDone = aggregator.sendResponse(reply);
                //     } else if (req.getCommand() != WorkerCommands.DESCRIBE) {
                //         int val = model.incrFailedInfReqs();
                //         logger.error("Number or consecutive unsuccessful inference {}", val);
                //         throw new WorkerInitializationException(
                //                 "Backend worker did not respond in given time");
                //     }
                //     totalDuration += duration;
                // } while (!jobDone);
                // logger.info("Backend response time: {}", totalDuration);

                // req = null;
                // double workerThreadTime =
                //         (System.currentTimeMillis() - wtStartTime) - totalDuration;
                // if (this.workerThreadTimeMetric != null) {
                //     try {
                //         this.workerThreadTimeMetric.addOrUpdate(
                //                 this.workerThreadTimeMetricDimensionValues, workerThreadTime);
                //     } catch (Exception e) {
                //         logger.error("Failed to update frontend metric WorkerThreadTime: ", e);
                //     }
                // }
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

    protected void connect() throws WorkerInitializationException, InterruptedException {
        if (!configManager.isDebug()) {
            lifeCycle.startWorker(port, getDeviceIds());
        }

        String modelName = model.getModelName();
        String modelVersion = model.getVersion();
        setState(WorkerState.WORKER_STARTED, HttpURLConnection.HTTP_OK);
        final int parallelLevel = model.getParallelLevel() > 0 ? model.getParallelLevel() : 1;
        final CountDownLatch latch = new CountDownLatch(parallelLevel);
        final int responseBufferSize = configManager.getMaxResponseSize();
        try {
            for (int i = 0; i < parallelLevel; i++) {
                Connector connector = new Connector(port + i);
                Bootstrap b = new Bootstrap();
                b.group(backendEventGroup)
                        .channel(connector.getClientChannel())
                        .handler(
                                new ChannelInitializer<Channel>() {
                                    @Override
                                    public void initChannel(Channel ch) {
                                        ChannelPipeline p = ch.pipeline();
                                        p.addLast(ENCODER);
                                        p.addLast(new ModelResponseDecoder(responseBufferSize));
                                        p.addLast(new WorkerHandler());
                                    }
                                });

                SocketAddress address = connector.getSocketAddress();
                logger.info("Connecting to: {}", address);
                backendChannel.add(b.connect(address).sync().channel());
                backendChannel
                        .get(i)
                        .closeFuture()
                        .addListener(
                                (ChannelFutureListener)
                                        future -> {
                                            latch.countDown();
                                            logger.info(
                                                    "{} Worker disconnected. {}",
                                                    getWorkerId(),
                                                    state);
                                            Thread thread = currentThread.getAndSet(null);
                                            if (thread != null) {
                                                thread.interrupt();
                                            }
                                        });
                backendChannel
                        .get(i)
                        .newSucceededFuture()
                        .addListener(
                                (ChannelFutureListener)
                                        future -> {
                                            // TODO:
                                            // use gpu, batch size in load model command
                                            if (latch.getCount() == 1) {
                                                RequestInput input =
                                                        new RequestInput(
                                                                UUID.randomUUID().toString());
                                                if (gpuId >= 0) {
                                                    input.addParameter(
                                                            new InputParameter(
                                                                    "gpu", String.valueOf(gpuId)));
                                                }

                                                Job job =
                                                        new RestJob(
                                                                null,
                                                                modelName,
                                                                modelVersion,
                                                                WorkerCommands.LOAD,
                                                                input);
                                                model.addJob(workerId, job);
                                            }
                                            latch.countDown();
                                        });
            }

            if (!latch.await(WORKER_TIMEOUT, TimeUnit.MINUTES)) {
                throw new WorkerInitializationException(
                        "Worker failed to initialize within " + WORKER_TIMEOUT + " mins");
            }
            running.set(true);
        } catch (Throwable t) {
            /* https://github.com/netty/netty/issues/2597 */
            if (t instanceof IOException) {
                throw new WorkerInitializationException("Failed to connect to worker.", t);
            }
            throw t;
        }
    }

    @ChannelHandler.Sharable
    protected class WorkerHandler extends SimpleChannelInboundHandler<ModelWorkerResponse> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, ModelWorkerResponse msg) {
            try {
                logger.info("MSG: {}", msg.getMessage());
                logger.info("Return code: {}", msg.getCode());
                aggregator.sendResponse(msg);
                if (!loadingFinished) {
                    if (msg.getCode() == 200) {
                        logger.info("Worker loaded the model successfully");
                        setState(WorkerState.WORKER_MODEL_LOADED, HttpURLConnection.HTTP_OK);
                        backoffIdx = 0;
                        loadingFinished = true;
                    } else {
                        setState(WorkerState.WORKER_ERROR, msg.getCode());
                    }
                }
            } catch (NullPointerException e) {
                logger.error(
                        "Failed to send response", e);
                throw new IllegalStateException("Message was empty");
            }
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            logger.error("Unknown exception", cause);
            if (cause instanceof OutOfMemoryError) {
                ModelWorkerResponse msg = new ModelWorkerResponse();
                msg.setCode(HttpURLConnection.HTTP_ENTITY_TOO_LARGE);
                msg.setMessage(cause.getMessage());
                if (!replies.offer(msg)) {
                    throw new IllegalStateException("Reply queue is full.");
                }
            }
            ctx.close();
        }
    }

}
