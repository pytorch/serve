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
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.SocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.metrics.IMetric;
import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
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
    private static final Logger loggerTelemetryMetrics =
            LoggerFactory.getLogger(ConfigManager.MODEL_SERVER_TELEMETRY_LOGGER);
    private static final int[] BACK_OFF = {
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597
    };
    private static final long WORKER_TIMEOUT = 2L;
    private static final ModelRequestEncoder ENCODER =
            new ModelRequestEncoder(ConfigManager.getInstance().getPreferDirectBuffer());
    private final IMetric workerThreadTimeMetric;
    private final IMetric workerLoadTimeMetric;
    private final List<String> workerThreadTimeMetricDimensionValues;
    private final List<String> workerLoadTimeMetricDimensionValues;
    private ConfigManager configManager;
    private EventLoopGroup backendEventGroup;
    private int port;
    private Model model;

    private ArrayList<Channel> backendChannel = new ArrayList<>();
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
    private WorkerState state;
    private WorkerLifeCycle lifeCycle;
    private int responseTimeout;
    private long recoveryStartTS; // 0: default value. no recovery needed, in healthy mode
    private BaseModelRequest req = null;

    public WorkerThread(
            ConfigManager configManager,
            EventLoopGroup backendEventGroup,
            int port,
            int gpuId,
            Model model,
            BatchAggregator aggregator,
            WorkerStateListener listener) {
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
        replies =
                new ArrayBlockingQueue<>(
                        model.getParallelLevel() > 0 ? model.getParallelLevel() : 1);
        this.workerThreadTimeMetric =
                MetricCache.getInstance().getMetricFrontend("WorkerThreadTime");
        this.workerLoadTimeMetric = MetricCache.getInstance().getMetricFrontend("WorkerLoadTime");
        this.workerThreadTimeMetricDimensionValues =
                Arrays.asList("Host", ConfigManager.getInstance().getHostName());
        this.workerLoadTimeMetricDimensionValues =
                Arrays.asList(getWorkerName(), "Host", ConfigManager.getInstance().getHostName());
    }

    public WorkerState getState() {
        return state;
    }

    public String getGpuUsage() {
        Process process;
        StringBuffer gpuUsage = new StringBuffer();
        if (gpuId >= 0) {
            try {
                // TODO : add a generic code to capture gpu details for different devices instead of
                // just NVIDIA
                ProcessBuilder pb =
                        new ProcessBuilder(
                                "nvidia-smi",
                                "-i",
                                String.valueOf(gpuId),
                                "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                                "--format=csv");

                // Start the process
                process = pb.start();
                process.waitFor();
                int exitCode = process.exitValue();
                if (exitCode != 0) {
                    gpuUsage.append("failed to obtained gpu usage");
                    InputStream error = process.getErrorStream();
                    for (int i = 0; i < error.available(); i++) {
                        logger.error("" + error.read());
                    }
                    return gpuUsage.toString();
                }
                InputStream stdout = process.getInputStream();
                BufferedReader reader =
                        new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
                String line;
                String[] headers = new String[3];
                Boolean firstLine = true;
                while ((line = reader.readLine()) != null) {
                    if (firstLine) {
                        headers = line.split(",");
                        firstLine = false;
                    } else {
                        String[] values = line.split(",");
                        StringBuffer sb = new StringBuffer("gpuId::" + gpuId + " ");
                        for (int i = 0; i < headers.length; i++) {
                            sb.append(headers[i] + "::" + values[i].strip());
                        }
                        gpuUsage.append(sb.toString());
                    }
                }
            } catch (Exception e) {
                gpuUsage.append("failed to obtained gpu usage");
                logger.error("Exception Raised : " + e.toString());
            }
        } else {
            gpuUsage.append("N/A");
        }

        return gpuUsage.toString();
    }

    public WorkerLifeCycle getLifeCycle() {
        return lifeCycle;
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
                    case STREAMPREDICT:
                    case STREAMPREDICT2:
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

    public String getWorkerId() {
        return workerId;
    }

    public long getMemory() {
        return memory;
    }

    public void setMemory(long memory) {
        this.memory = memory;
    }

    private void connect() throws WorkerInitializationException, InterruptedException {
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
        aggregator.shutdown();
        setState(WorkerState.WORKER_SCALED_DOWN, HttpURLConnection.HTTP_OK);
        for (int i = 0;
                backendChannel.size() > 0
                        && i < (model.getParallelLevel() > 0 ? model.getParallelLevel() : 1);
                i++) {
            if (backendChannel.get(i) != null) {
                backendChannel.get(i).close();
            }
        }
        backendChannel.clear();
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

    public void setState(WorkerState newState, int status) {
        listener.notifyChangeState(
                model.getModelVersionName().getVersionedModelName(), newState, status);
        logger.debug("{} State change {} -> {}", getWorkerName(), state, newState);
        long currentTS = System.currentTimeMillis();
        long timeTaken = currentTS - startTime;
        if (state != WorkerState.WORKER_SCALED_DOWN) {
            // Don't update the state if it was terminated on purpose.. Scaling in..
            this.state = newState;
        }

        if (state == WorkerState.WORKER_MODEL_LOADED) {
            if (this.workerLoadTimeMetric != null) {
                try {
                    this.workerLoadTimeMetric.addOrUpdate(
                            this.workerLoadTimeMetricDimensionValues, timeTaken);
                } catch (Exception e) {
                    logger.error("Failed to update frontend metric WorkerLoadTime: ", e);
                }
            }
            if (recoveryStartTS > 0) {
                logger.info("Auto recovery succeeded, reset recoveryStartTS");
                recoveryStartTS = 0;
            }
        } else if (state == WorkerState.WORKER_STOPPED) {
            if (recoveryStartTS == 0) {
                recoveryStartTS = currentTS;
                logger.info("Auto recovery start timestamp: {}", recoveryStartTS);
            } else {
                logger.warn("Auto recovery failed again");
            }
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
        aggregator.startEventDispatcher();

        manager.getScheduler()
                .schedule(() -> manager.submitTask(this), BACK_OFF[backoffIdx], TimeUnit.SECONDS);
        logger.info("Retry worker: {} in {} seconds.", workerId, BACK_OFF[backoffIdx]);
    }

    private String getDeviceIds() {
        List<Integer> deviceIds;
        if (gpuId == -1 || model.getParallelLevel() == 0) {
            return null;
        } else if (model.isHasCfgDeviceIds()) {
            return model.getDeviceIds().subList(gpuId, gpuId + model.getParallelLevel()).stream()
                    .map(String::valueOf)
                    .collect(Collectors.joining(","));
        } else {
            deviceIds = new ArrayList<>(model.getParallelLevel());
            for (int i = gpuId; i < gpuId + model.getParallelLevel(); i++) {
                deviceIds.add(i);
            }
            return deviceIds.stream().map(String::valueOf).collect(Collectors.joining(","));
        }
    }

    public boolean isHealthy() {
        if (recoveryStartTS == 0
                || (System.currentTimeMillis() - recoveryStartTS)
                        < model.getMaxRetryTimeoutInMill()) {
            return true;
        }
        return false;
    }

    @ChannelHandler.Sharable
    private class WorkerHandler extends SimpleChannelInboundHandler<ModelWorkerResponse> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, ModelWorkerResponse msg) {
            try {
                replies.offer(msg, responseTimeout, TimeUnit.SECONDS);
            } catch (InterruptedException | NullPointerException e) {
                logger.error(
                        "Failed to offer reply, responseTimeout:" + responseTimeout + "sec", e);
                throw new IllegalStateException("Reply queue is full.");
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

    private boolean isTensorParallelRequest(WorkerCommands workerCmd) {
        switch (workerCmd) {
            case PREDICT:
            case STREAMPREDICT:
            case STREAMPREDICT2:
                if (model.hasTensorParallel()) {
                    return true;
                }
                return false;
            default:
                return false;
        }
    }

    private boolean isLoadRequest(WorkerCommands workerCmd) {
        return workerCmd == WorkerCommands.LOAD;
    }

    private int getRepeats(WorkerCommands workerCmd) {
        if (isLoadRequest(workerCmd) || isTensorParallelRequest(workerCmd)) {
            // broadcast the command to all ranks
            return Math.max(1, model.getParallelLevel());
        }

        return 1;
    }
}
