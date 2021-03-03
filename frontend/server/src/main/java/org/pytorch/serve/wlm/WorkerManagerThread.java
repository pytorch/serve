package org.pytorch.serve.wlm;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.SocketAddress;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.metrics.Dimension;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.SharedNamedPipeUtils;
import org.pytorch.serve.util.codec.ModelRequestEncoder;
import org.pytorch.serve.util.codec.ModelResponseDecoder;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerManagerThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(WorkerManagerThread.class);
    private static final org.apache.log4j.Logger loggerTsMetrics =
            org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);

    private Metric workerLoadTime;

    private static final int[] BACK_OFF = {
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597
    };

    private static final long WORKER_TIMEOUT = 2L;
    private static final ModelRequestEncoder ENCODER =
            new ModelRequestEncoder(ConfigManager.getInstance().getPreferDirectBuffer());

    private ConfigManager configManager;
    private EventLoopGroup backendEventGroup;
    private int port;
    private Model model;

    private Channel backendChannel;
    private AtomicBoolean running = new AtomicBoolean(true);

    private int backoffIdx;

    private BatchAggregator aggregator;
    private WorkerManagerStateListener listener;
    private ArrayBlockingQueue<ModelWorkerResponse> replies;
    private long memory;
    private long startTime;
    private AtomicReference<Thread> currentThread = new AtomicReference<>();
    private String workerId;
    private int gpuId;

    private WorkerManagerState state;

    private WorkerManagerLifeCycle lifeCycle;

    public WorkerManagerState getState() {
        return state;
    }

    public WorkerManagerLifeCycle getLifeCycle() {
        return lifeCycle;
    }

    public WorkerManagerThread(
            ConfigManager configManager,
            EventLoopGroup backendEventGroup,
            int port,
            Model model,
            BatchAggregator aggregator,
            WorkerManagerStateListener listener) {

        this.workerId = String.valueOf(port); // Unique across all workers.
        this.configManager = configManager;
        this.backendEventGroup = backendEventGroup;
        this.port = port;
        this.model = model;
        this.aggregator = aggregator;
        this.gpuId = gpuId;
        this.listener = listener;
        startTime = System.currentTimeMillis();
        lifeCycle = new WorkerManagerLifeCycle(configManager, model);
        replies = new ArrayBlockingQueue<>(1);
        workerLoadTime =
                new Metric(
                        getWorkerName(),
                        String.valueOf(System.currentTimeMillis()),
                        "ms",
                        ConfigManager.getInstance().getHostName(),
                        new Dimension("Level", "Host"));
    }

    @Override
    public void run() {
        int responseTimeout = model.getResponseTimeout();
        Thread thread = Thread.currentThread();
        thread.setName(getWorkerName());
        currentThread.set(thread);
        BaseModelRequest req = null;
        int status = HttpURLConnection.HTTP_INTERNAL_ERROR;

        try {
            connect();

            while (isRunning()) {
                req = aggregator.getCtrlRequest(workerId, state);

                if (req == null) {
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        logger.info("Waiting to ctrl requests ..");
                    }
                    continue;
                }

                long wtStartTime = System.currentTimeMillis();
                backendChannel.writeAndFlush(req).sync();

                long begin = System.currentTimeMillis();
                ModelWorkerResponse reply = replies.poll(responseTimeout, TimeUnit.SECONDS);

                long duration = System.currentTimeMillis() - begin;
                logger.info("Backend response time: {}", duration);

                if (reply != null) {
                    aggregator.sendResponse(reply);
                } else {
                    int val = model.incrFailedInfReqs();
                    logger.error("Number or consecutive unsuccessful inference {}", val);
                    throw new WorkerInitializationException(
                            "Backend worker did not respond in given time");
                }
                switch (req.getCommand()) {
                    case LOAD:
                        if (reply.getCode() == 200) {
                            setState(
                                    WorkerManagerState.WORKER_MODEL_LOADED,
                                    HttpURLConnection.HTTP_OK);
                            backoffIdx = 0;
                        } else {
                            setState(WorkerManagerState.WORKER_ERROR, reply.getCode());
                            status = reply.getCode();
                        }
                        break;
                    case SCALE_UP:
                        if (reply.getCode() == 200) {
                            setState(
                                    WorkerManagerState.WORKER_MODEL_LOADED,
                                    HttpURLConnection.HTTP_OK);
                            backoffIdx = 0;
                        } else {
                            setState(WorkerManagerState.WORKER_MODEL_LOADED, reply.getCode());
                            status = reply.getCode();
                        }
                        break;
                    case SCALE_DOWN:
                        if (reply.getCode() == 200) {
                            setState(
                                    WorkerManagerState.WORKER_SCALED_DOWN,
                                    HttpURLConnection.HTTP_OK);
                            backoffIdx = 0;
                        } else {
                            setState(WorkerManagerState.WORKER_ERROR, reply.getCode());
                            status = reply.getCode();
                        }
                        break;
                    default:
                        break;
                }
                req = null;
                String workerThreadTime =
                        String.valueOf(((System.currentTimeMillis() - wtStartTime) - duration));
                loggerTsMetrics.info(
                        new Metric(
                                "WorkerThreadTime",
                                workerThreadTime,
                                "ms",
                                ConfigManager.getInstance().getHostName(),
                                new Dimension("Level", "Host")));
            }
        } catch (InterruptedException e) {
            logger.debug("System state is : " + state);
            if (state == WorkerManagerState.WORKER_SCALED_DOWN
                    || state == WorkerManagerState.WORKER_STOPPED) {
                logger.debug("Shutting down the thread .. Scaling down.");
            } else {
                logger.debug(
                        "Backend worker monitoring thread interrupted or backend worker process died.",
                        e);
            }
        } catch (WorkerInitializationException e) {
            logger.error("Backend worker error", e);
        } catch (OutOfMemoryError oom) {
            logger.error("Out of memory error when creating workers", oom);
            status = HttpURLConnection.HTTP_INTERNAL_ERROR;
        } catch (Throwable t) {
            logger.warn("Backend worker thread exception.", t);
        } finally {
            // WorkerThread is running in thread pool, the thread will be assigned to next
            // Runnable once this worker is finished. If currentThread keep holding the reference
            // of the thread, currentThread.interrupt() might kill next worker.
            backendChannel.disconnect();
            currentThread.set(null);
            Integer exitValue = lifeCycle.getExitValue();

            if (exitValue != null && exitValue == 137) {
                status = HttpURLConnection.HTTP_INTERNAL_ERROR;
            }

            if (req != null) {
                aggregator.sendError(req, "Worker died.", status);
            }
            setState(WorkerManagerState.WORKER_STOPPED, status);
            lifeCycle.exit();
            retry();
        }
    }

    public String getWorkerId() {
        return workerId;
    }

    public long getMemory() {
        return memory;
    }

    public void scaleUp(int port) {

        RequestInput input = new RequestInput(UUID.randomUUID().toString());

        Connector connector = new Connector(port);

        input.addParameter(new InputParameter("sock_type", connector.getSocketType()));

        if (connector.isUds()) {
            input.addParameter(new InputParameter("sock_name", connector.getSocketPath()));
        } else {
            input.addParameter(
                    new InputParameter("port", String.valueOf(connector.getSocketPath())));
        }

        input.addParameter(
                new InputParameter(
                        "fifo_path",
                        SharedNamedPipeUtils.getSharedNamedPipePath(String.valueOf(port))));

        Job job =
                new RestJob(
                        null,
                        model.getModelName(),
                        model.getVersion(),
                        WorkerCommands.SCALE_UP,
                        input);
        model.addJob(workerId, job);
    }

    public void scaleDown(int port) {

        RequestInput input = new RequestInput(UUID.randomUUID().toString());

        Connector connector = new Connector(port);

        input.addParameter(new InputParameter("port", String.valueOf(connector.getSocketPath())));

        Job job =
                new RestJob(
                        null,
                        model.getModelName(),
                        model.getVersion(),
                        WorkerCommands.SCALE_DOWN,
                        input);
        model.addJob(workerId, job);
    }

    public void setMemory(long memory) {
        this.memory = memory;
    }

    private void connect() throws WorkerInitializationException, InterruptedException {
        if (!configManager.isDebug()) {
            lifeCycle.startWorkerManager(port);
        }

        String modelName = model.getModelName();
        String modelVersion = model.getVersion();
        setState(WorkerManagerState.WORKER_STARTED, HttpURLConnection.HTTP_OK);
        final CountDownLatch latch = new CountDownLatch(1);

        final int responseBufferSize = configManager.getMaxResponseSize();
        try {
            Connector connector = new Connector(port);
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
            backendChannel = b.connect(address).sync().channel();
            backendChannel
                    .closeFuture()
                    .addListener(
                            (ChannelFutureListener)
                                    future -> {
                                        latch.countDown();
                                        logger.info(
                                                "{} Worker disconnected. {}", getWorkerId(), state);
                                        Thread thread = currentThread.getAndSet(null);
                                        if (thread != null) {
                                            thread.interrupt();
                                        }
                                    });

            backendChannel
                    .newSucceededFuture()
                    .addListener(
                            (ChannelFutureListener)
                                    future -> {
                                        // TODO:
                                        // use gpu, batch size in load model command
                                        RequestInput input =
                                                new RequestInput(UUID.randomUUID().toString());

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
                                        model.addFirst(workerId, job);
                                        latch.countDown();
                                    });

            if (!latch.await(WORKER_TIMEOUT, TimeUnit.MINUTES)) {
                throw new WorkerInitializationException(
                        "Worker failed to initialize within " + WORKER_TIMEOUT + " mins");
            }
            running.set(true);
        } catch (Throwable t) {
            // https://github.com/netty/netty/issues/2597
            if (t instanceof IOException) {
                throw new WorkerInitializationException("Failed to connect to worker.", t);
            }
            throw t;
        }
    }

    public boolean isRunning() {
        return running.get();
    }

    public long getStartTime() {
        return startTime;
    }

    public int getPid() {
        return lifeCycle.getPid();
    }

    public void shutdown() {
        running.set(false);
        setState(WorkerManagerState.WORKER_SCALED_DOWN, HttpURLConnection.HTTP_OK);
        if (backendChannel != null) {
            backendChannel.close();
        }
        lifeCycle.terminateIOStreams();
        Thread thread = currentThread.getAndSet(null);
        if (thread != null) {
            thread.interrupt();
            aggregator.sendError(
                    null, "Worker scaled down.", HttpURLConnection.HTTP_INTERNAL_ERROR);

            model.removeJobQueue(workerId);
        }
    }

    private String getWorkerName() {
        String modelName = model.getModelVersionName().getVersionedModelName();
        return "W-" + port + '-' + modelName;
    }

    public void setState(WorkerManagerState newState, int status) {
        listener.notifyChangeState(
                model.getModelVersionName().getVersionedModelName(), newState, status);
        logger.debug("{} State change {} -> {}", getWorkerName(), state, newState);
        long timeTaken = System.currentTimeMillis() - startTime;
        if (state != WorkerManagerState.WORKER_SCALED_DOWN) {
            // Don't update the state if it was terminated on purpose.. Scaling in..
            this.state = newState;
        }
        if (state == WorkerManagerState.WORKER_MODEL_LOADED) {
            workerLoadTime.setValue(String.valueOf(timeTaken));
            workerLoadTime.setTimestamp(
                    String.valueOf(TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis())));
            loggerTsMetrics.info(workerLoadTime);
        }
    }

    public void retry() {
        if (state == WorkerManagerState.WORKER_SCALED_DOWN) {
            logger.debug("Worker terminated due to scale-down call.");
            return;
        }

        ModelManager manager = ModelManager.getInstance();

        if (backoffIdx < BACK_OFF.length - 1) {
            ++backoffIdx;
        }

        manager.getScheduler()
                .schedule(() -> manager.submitTask(this), BACK_OFF[backoffIdx], TimeUnit.SECONDS);
        logger.info("Retry worker: {} in {} seconds.", workerId, BACK_OFF[backoffIdx]);
    }

    @ChannelHandler.Sharable
    private class WorkerHandler extends SimpleChannelInboundHandler<ModelWorkerResponse> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, ModelWorkerResponse msg) {
            if (!replies.offer(msg)) {
                throw new IllegalStateException("Reply queue is full.");
            }
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            logger.error("Unknown exception", cause);
            if (cause instanceof OutOfMemoryError) {
                NettyUtils.sendError(ctx, HttpResponseStatus.INSUFFICIENT_STORAGE, cause);
            }
            ctx.close();
        }
    }
}
