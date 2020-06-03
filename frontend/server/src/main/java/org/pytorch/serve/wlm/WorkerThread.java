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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.SocketAddress;
import java.nio.channels.Channels;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.pytorch.serve.metrics.Dimension;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.codec.ModelRequestEncoder;
import org.pytorch.serve.util.codec.ModelResponseDecoder;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);
    private static final org.apache.log4j.Logger loggerTsMetrics =
            org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);

    private Metric workerLoadTime;

    private static final int[] BACK_OFF = {
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597
    };

    private static final long WORKER_TIMEOUT =
            ConfigManager.getInstance().isDebug() ? Long.MAX_VALUE : 2L;
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
    private WorkerStateListener listener;
    private ArrayBlockingQueue<ModelWorkerResponse> replies;
    private int gpuId;
    private long memory;
    private long startTime;
    private AtomicReference<Thread> currentThread = new AtomicReference<>();
    private String workerId;

    private String threadName;
    private BaseModelRequest req;

    private WorkerState state;
    private boolean serverThread;

    private WorkerLifeCycle lifeCycle;

    public WorkerState getState() {
        return state;
    }

    public WorkerLifeCycle getLifeCycle() {
        return lifeCycle;
    }

    public WorkerThread(
            ConfigManager configManager,
            EventLoopGroup backendEventGroup,
            int port,
            int gpuId,
            Model model,
            BatchAggregator aggregator,
            WorkerStateListener listener,
            int threadNumber,
            boolean serverThread) {
        this.workerId = String.valueOf(port); // Unique across all workers.
        this.configManager = configManager;
        this.backendEventGroup = backendEventGroup;
        this.port = port;
        this.model = model;
        this.aggregator = aggregator;
        this.gpuId = gpuId;
        this.listener = listener;
        startTime = System.currentTimeMillis();
        lifeCycle = new WorkerLifeCycle(configManager, model);
        replies = new ArrayBlockingQueue<>(1);
        this.serverThread = serverThread;
        this.threadName =
                !serverThread
                        ? getWorkerName() + '-' + threadNumber
                        : "BackendServer-" + model.getModelVersionName().getVersionedModelName();
        workerLoadTime =
                new Metric(
                        getWorkerName(),
                        String.valueOf(System.currentTimeMillis()),
                        "ms",
                        configManager.getHostName(),
                        new Dimension("Level", "Host"));
    }

    private void runWorker()
            throws WorkerInitializationException, InterruptedException, FileNotFoundException {
        int responseTimeout = model.getResponseTimeout();
        while (isRunning()) {
            req = aggregator.getRequest(backendChannel.id().asLongText(), state);
            backendChannel.writeAndFlush(req).sync();
            long begin = System.currentTimeMillis();
            // TODO: Change this to configurable param
            ModelWorkerResponse reply = replies.poll(responseTimeout, TimeUnit.MINUTES);
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
                case PREDICT:
                    model.resetFailedInfReqs();
                    break;
                case LOAD:
                    String message = reply.getMessage();
                    String tmpdir = System.getProperty("java.io.tmpdir");
                    RandomAccessFile out =
                            new RandomAccessFile(
                                    tmpdir + '/' + backendChannel.id().asLongText() + "-stdout",
                                    "rw");
                    RandomAccessFile err =
                            new RandomAccessFile(
                                    tmpdir + '/' + backendChannel.id().asLongText() + "-stderr",
                                    "rw");
                    if (reply.getCode() == 200) {
                        setState(WorkerState.WORKER_MODEL_LOADED, HttpResponseStatus.OK);
                        lifeCycle.setPid(
                                Integer.parseInt(
                                        message.substring(
                                                message.indexOf("[PID]:") + 6, message.length())));
                        lifeCycle.attachIOStreams(
                                threadName,
                                Channels.newInputStream(out.getChannel()),
                                Channels.newInputStream(err.getChannel()));
                        backoffIdx = 0;
                    } else {
                        setState(
                                WorkerState.WORKER_ERROR,
                                HttpResponseStatus.valueOf(reply.getCode()));
                    }
                    break;
                case UNLOAD:
                case STATS:
                default:
                    break;
            }
            req = null;
        }
    }

    @Override
    public void run() {
        Process process = null;
        Thread thread = Thread.currentThread();
        thread.setName(getWorkerName());
        currentThread.set(thread);
        HttpResponseStatus status = HttpResponseStatus.INTERNAL_SERVER_ERROR;

        try {
            if (!serverThread) {
                connect();
                runWorker();
            } else {
                // TODO: Move this logic to a seperate ServerThread class
                // This is server thread and shouldn't come out as long as process exists in CPU.
                model.setPort(port);
                lifeCycle.startBackendServer(port);
                setState(WorkerState.WORKER_MODEL_LOADED, HttpResponseStatus.OK);
                process = lifeCycle.getProcess();
                process.waitFor();
            }
        } catch (InterruptedException e) {
            logger.debug("System state is : " + state);
            if (state == WorkerState.WORKER_SCALED_DOWN || state == WorkerState.WORKER_STOPPED) {
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
            status = HttpResponseStatus.INSUFFICIENT_STORAGE;
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
                status = HttpResponseStatus.INSUFFICIENT_STORAGE;
            }

            if (!serverThread && req != null) {
                aggregator.sendError(req, "Worker died.", status);
            } else if (serverThread) {
                model.setPort(-1);
                if (process != null && process.isAlive()) {
                    process.destroyForcibly();
                    try {
                        process.waitFor(1, TimeUnit.SECONDS);
                    } catch (InterruptedException e) {
                        logger.warn(
                                "WorkerThread interrupted during waitFor, possible async resource cleanup.");
                    }
                }
                aggregator.sendError(req, "Worker died.", status);
            }
            setState(WorkerState.WORKER_STOPPED, status);
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

    public void setMemory(long memory) {
        this.memory = memory;
    }

    private void connect()
            throws WorkerInitializationException, InterruptedException, FileNotFoundException {
        if (!this.serverThread && (model.getPort() == -1)) {
            throw new WorkerInitializationException("Backend server is not running");
        }

        String modelName = model.getModelName();
        String modelVersion = model.getVersion();
        setState(WorkerState.WORKER_STARTED, HttpResponseStatus.OK);
        final CountDownLatch latch = new CountDownLatch(1);

        final int responseBufferSize = configManager.getMaxResponseSize();
        try {
            Connector connector = new Connector(model.getPort());
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
                                                new Job(
                                                        null,
                                                        modelName,
                                                        modelVersion,
                                                        WorkerCommands.LOAD,
                                                        input);
                                        model.addJob(backendChannel.id().asLongText(), job);
                                        latch.countDown();
                                    });

            if (!latch.await(WORKER_TIMEOUT, TimeUnit.MINUTES)) {
                throw new WorkerInitializationException(
                        "Worker failed to initialize within " + WORKER_TIMEOUT + " mins");
            }
            workerId = workerId + "-" + backendChannel.id().asLongText();
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

    public int getGpuId() {
        return gpuId;
    }

    public long getStartTime() {
        return startTime;
    }

    public int getPid() {
        return lifeCycle.getPid();
    }

    public void shutdown() {
        running.set(false);
        setState(WorkerState.WORKER_SCALED_DOWN, HttpResponseStatus.OK);
        if (backendChannel != null) {
            model.removeJobQueue(backendChannel.id().asLongText());
            backendChannel.close();
        }
        lifeCycle.terminateIOStreams();
        Thread thread = currentThread.getAndSet(null);
        if (thread != null) {
            thread.interrupt();
            aggregator.sendError(
                    null, "Worker scaled down.", HttpResponseStatus.INTERNAL_SERVER_ERROR);

            model.removeJobQueue(workerId);
        }
    }

    public boolean isServerThread() {
        return serverThread;
    }

    private String getWorkerName() {
        String modelName = model.getModelVersionName().getVersionedModelName();
        return "W-" + port + '-' + modelName;
    }

    public void setState(WorkerState newState, HttpResponseStatus status) {
        listener.notifyChangeState(
                model.getModelVersionName().getVersionedModelName(), newState, status);
        logger.debug("{} State change {} -> {}", getWorkerName(), state, newState);
        long timeTaken = System.currentTimeMillis() - startTime;
        if (state != WorkerState.WORKER_SCALED_DOWN) {
            // Don't update the state if it was terminated on purpose.. Scaling in..
            this.state = newState;
        }
        if (state == WorkerState.WORKER_MODEL_LOADED) {
            workerLoadTime.setValue(String.valueOf(timeTaken));
            workerLoadTime.setTimestamp(
                    String.valueOf(TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis())));
            loggerTsMetrics.info(workerLoadTime);
        }
    }

    public void retry() {
        if (state == WorkerState.WORKER_SCALED_DOWN) {
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
