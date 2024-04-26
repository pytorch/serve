package org.pytorch.serve.wlm;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.pytorch.serve.archive.model.ModelConfig;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.pytorch.serve.util.messages.EnvironmentUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerLifeCycle {

    private static final Logger logger = LoggerFactory.getLogger(WorkerLifeCycle.class);
    private static final Pattern PID_LOG_PATTERN = Pattern.compile(".*\\[PID\\](\\d+)$");
    private static final String METRIC_LOG_START_SUBSTRING = "[METRICS]";

    private ConfigManager configManager;
    private ModelManager modelManager = ModelManager.getInstance();
    private Model model;
    private int pid = -1;
    private Process process;
    private CountDownLatch latch;
    private boolean success;
    private Connector connector;
    private ReaderThread errReader;
    private ReaderThread outReader;
    private int numWorker;
    private int currNumRunningWorkers;

    public WorkerLifeCycle(ConfigManager configManager, Model model) {
        this.configManager = configManager;
        this.model = model;
        this.numWorker = model.getMinWorkers();
        this.currNumRunningWorkers = modelManager.getNumRunningWorkers(model.getModelVersionName());
    }

    public Process getProcess() {
        return process;
    }

    public ArrayList<String> launcherArgsToList(String launcherArgs) {
        ArrayList<String> arrlist = new ArrayList<String>();
        arrlist.add("-m");
        arrlist.add("torch.backends.xeon.run_cpu");

        if (launcherArgs != null && launcherArgs.length() > 1) {
            String[] argarray = launcherArgs.split(" ");
            for (int i = 0; i < argarray.length; i++) {
                arrlist.add(argarray[i]);
            }
        }
        return arrlist;
    }

    public boolean isLauncherAvailable(String launcherArgs)
            throws WorkerInitializationException, InterruptedException {
        boolean launcherAvailable = false;

        ArrayList<String> cmd = new ArrayList<String>();
        cmd.add("python");
        ArrayList<String> args = launcherArgsToList(launcherArgs);
        cmd.addAll(args);
        cmd.add("--no_python");
        // try launching dummy command to check launcher availability
        String dummyCmd = "hostname";
        cmd.add(dummyCmd);

        String[] cmdList = new String[cmd.size()];
        cmdList = cmd.toArray(cmdList);

        logger.debug("launcherAvailable cmdline: {}", cmd.toString());

        try {
            Process processLauncher = Runtime.getRuntime().exec(cmdList);
            int ret = processLauncher.waitFor();
            launcherAvailable = (ret == 0);
        } catch (IOException | InterruptedException e) {
            throw new WorkerInitializationException("Failed to start launcher", e);
        }
        return launcherAvailable;
    }

    public void startWorker(int port, String deviceIds)
            throws WorkerInitializationException, InterruptedException {
        switch (model.getRuntimeType()) {
            case LSP:
                logger.info("LSP startWorker");
                startWorkerCPP(port, "LSP", deviceIds);
                break;
            default:
                startWorkerPython(port, deviceIds);
                break;
        }
    }

    private void startWorkerPython(int port, String deviceIds)
            throws WorkerInitializationException, InterruptedException {
        File workingDir = new File(configManager.getModelServerHome());
        File modelPath;
        setPort(port);
        try {
            modelPath = model.getModelDir();
            // Test if modelPath is valid
            modelPath.getCanonicalFile();
        } catch (IOException e) {
            throw new WorkerInitializationException("Failed get TS home directory", e);
        }

        ArrayList<String> argl = new ArrayList<>();
        ArrayList<String> envp = new ArrayList<>();
        envp.addAll(
                Arrays.asList(
                        EnvironmentUtils.getEnvString(
                                workingDir.getAbsolutePath(),
                                modelPath.getAbsolutePath(),
                                model.getModelArchive().getManifest().getModel().getHandler())));

        if (model.getParallelLevel() > 0) {
            attachRunner(argl, envp, port, deviceIds);
        } else if (model.getParallelLevel() == 0) {
            argl.add(EnvironmentUtils.getPythonRunTime(model));
        }

        if (configManager.isCPULauncherEnabled()) {
            String launcherArgs = configManager.getCPULauncherArgs();
            boolean launcherAvailable = isLauncherAvailable(launcherArgs);
            if (launcherAvailable) {
                ArrayList<String> args = launcherArgsToList(launcherArgs);
                argl.addAll(args);

                // multi-worker core pinning
                if (this.numWorker > 1) {
                    argl.add("--ninstances");
                    argl.add(String.valueOf(this.numWorker));
                    argl.add("--rank");
                    // instance_idx is 0-indexed
                    argl.add(String.valueOf(this.currNumRunningWorkers));
                }

            } else {
                logger.warn(
                        "torch.backends.xeon.run_cpu is not available. Proceeding without worker core pinning. For better performance, please make sure torch.backends.xeon.run_cpu is available.");
            }
        }

        argl.add(new File(workingDir, "ts/model_service_worker.py").getAbsolutePath());
        argl.add("--sock-type");
        argl.add(connector.getSocketType());
        argl.add(connector.isUds() ? "--sock-name" : "--port");
        argl.add(connector.getSocketPath());

        argl.add("--metrics-config");
        argl.add(configManager.getMetricsConfigPath());

        try {
            latch = new CountDownLatch(model.getParallelLevel() > 0 ? model.getParallelLevel() : 1);

            String[] args = argl.toArray(new String[argl.size()]);
            String[] envs = envp.toArray(new String[envp.size()]);
            logger.debug("Worker cmdline: {}", argl.toString());

            synchronized (this) {
                process = Runtime.getRuntime().exec(args, envs, modelPath);

                String threadName =
                        "W-" + port + '-' + model.getModelVersionName().getVersionedModelName();
                errReader = new ReaderThread(threadName, process.getErrorStream(), true, this);
                outReader = new ReaderThread(threadName, process.getInputStream(), false, this);
                errReader.start();
                outReader.start();
            }

            if (latch.await(2, TimeUnit.MINUTES)) {
                if (!success) {
                    throw new WorkerInitializationException("Backend stream closed.");
                }
                return;
            }
            throw new WorkerInitializationException("Backend worker startup time out.");
        } catch (IOException e) {
            throw new WorkerInitializationException("Failed start worker process", e);
        } finally {
            if (!success) {
                exit();
            }
        }
    }

    private void startWorkerCPP(int port, String runtimeType, String deviceIds)
            throws WorkerInitializationException, InterruptedException {
        File workingDir = new File(configManager.getModelServerHome());
        File modelPath;
        setPort(port);
        try {
            modelPath = model.getModelDir().getCanonicalFile();
        } catch (IOException e) {
            throw new WorkerInitializationException("Failed get TS home directory", e);
        }

        ArrayList<String> argl = new ArrayList<String>();

        File cppBackendBin = new File(workingDir, "ts/cpp/bin/model_worker_socket");
        File cppBackendLib = new File(workingDir, "ts/cpp/lib");
        if (!cppBackendBin.exists()) {
            throw new WorkerInitializationException("model_worker_socket not found");
        }
        if (!cppBackendLib.exists()) {
            throw new WorkerInitializationException("model_worker cpp library not found");
        }

        argl.add(cppBackendBin.getAbsolutePath());
        argl.add("--sock_type");
        argl.add(connector.getSocketType());
        argl.add(connector.isUds() ? "--sock_name" : "--port");
        argl.add(connector.getSocketPath());
        argl.add("--runtime_type");
        argl.add(runtimeType);
        argl.add("--model_dir");
        argl.add(modelPath.getAbsolutePath());
        if (ConfigManager.getInstance().getTsCppLogConfig() != null) {
            argl.add("--logger_config_path");
            argl.add(ConfigManager.getInstance().getTsCppLogConfig());
        }
        argl.add("--metrics_config_path");
        argl.add(configManager.getMetricsConfigPath());

        String[] envp = EnvironmentUtils.getCppEnvString(cppBackendLib.getAbsolutePath());

        try {
            latch = new CountDownLatch(1);

            String[] args = argl.toArray(new String[argl.size()]);
            logger.debug("Worker cmdline: {}", argl.toString());

            synchronized (this) {
                process = Runtime.getRuntime().exec(args, envp, modelPath);

                String threadName =
                        "W-" + port + '-' + model.getModelVersionName().getVersionedModelName();
                errReader = new ReaderThread(threadName, process.getErrorStream(), true, this);
                outReader = new ReaderThread(threadName, process.getInputStream(), false, this);
                errReader.start();
                outReader.start();
            }

            if (latch.await(2, TimeUnit.MINUTES)) {
                if (!success) {
                    throw new WorkerInitializationException("Backend stream closed.");
                }
                return;
            }
            throw new WorkerInitializationException("Backend worker startup time out.");
        } catch (IOException e) {
            throw new WorkerInitializationException("Failed start worker process", e);
        } finally {
            if (!success) {
                exit();
            }
        }
    }

    private void attachRunner(
            ArrayList<String> argl, List<String> envp, int port, String deviceIds) {
        envp.add("LOGLEVEL=INFO");
        if (deviceIds != null) {
            envp.add("CUDA_VISIBLE_DEVICES=" + deviceIds);
        }
        ModelConfig.TorchRun torchRun = model.getModelArchive().getModelConfig().getTorchRun();
        envp.add(String.format("OMP_NUM_THREADS=%d", torchRun.getOmpNumberThreads()));
        argl.add("torchrun");
        argl.add("--nnodes");
        argl.add(String.valueOf(torchRun.getNnodes()));
        argl.add("--nproc-per-node");
        argl.add(String.valueOf(torchRun.getNprocPerNode()));
        argl.add("--log-dir");
        argl.add(ConfigManager.getInstance().getTorchRunLogDir());
        argl.add("--rdzv-backend");
        argl.add(torchRun.getRdzvBackend());
        if (torchRun.getRdzvEndpoint() != null) {
            argl.add("--rdzv-endpoint");
            argl.add(torchRun.getRdzvEndpoint());
        }
        argl.add("--rdzv-id");
        argl.add(String.format("%s_%d", model.getModelName(), port));
        if (torchRun.getMasterAddr() != null) {
            argl.add("--master-addr");
            argl.add(torchRun.getMasterAddr());
            argl.add("--master-port");
            argl.add(String.valueOf(torchRun.getMasterPort()));
        }
        argl.add("--max-restarts");
        argl.add(String.valueOf(1));
    }

    public synchronized void exit() {
        if (process != null) {
            process.destroyForcibly();
            connector.clean();
        }
    }

    public synchronized Integer getExitValue() {
        if (process != null && !process.isAlive()) {
            return process.exitValue();
        }
        return null;
    }

    public void setSuccess(boolean success) {
        this.success = success;
        latch.countDown();
    }

    public synchronized int getPid() {
        return pid;
    }

    public synchronized void setPid(int pid) {
        this.pid = pid;
    }

    private synchronized void setPort(int port) {
        connector = new Connector(port);
    }

    private static final class ReaderThread extends Thread {
        private static final Pattern METRIC_PATTERN =
                Pattern.compile("^(INFO > )?(\\[METRICS])(.*)");
        // TODO: Fix logging format in cpp backend
        private static final Pattern WORKER_START_PATTERN =
                Pattern.compile("(.*)(INFO > )?(Torch worker started.)$");
        private static final Pattern WORKER_PID_PATTERN =
                Pattern.compile("^(INFO > )?(\\[PID])(\\d+)$");
        private static final Logger loggerModelOutput =
                LoggerFactory.getLogger(ConfigManager.MODEL_LOGGER);
        private final MetricCache metricCache;
        private InputStream is;
        private boolean error;
        private WorkerLifeCycle lifeCycle;
        private AtomicBoolean isRunning = new AtomicBoolean(true);

        public ReaderThread(String name, InputStream is, boolean error, WorkerLifeCycle lifeCycle) {
            super(name + (error ? "-stderr" : "-stdout"));
            this.is = is;
            this.error = error;
            this.lifeCycle = lifeCycle;
            this.metricCache = MetricCache.getInstance();
        }

        @Override
        public void run() {
            try (Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name())) {
                while (scanner.hasNextLine()) {
                    String result = scanner.nextLine();
                    Matcher matcher = METRIC_PATTERN.matcher(result);
                    if (matcher.matches()) {
                        logger.info("result={}, pattern={}", result, matcher.group(2));

                        Metric parsedMetric = Metric.parse(matcher.group(3));
                        if (parsedMetric == null) {
                            logger.error("Failed to parse metrics line: \"{}\".", result);
                            continue;
                        }

                        try {
                            if (this.metricCache.getMetricBackend(parsedMetric.getMetricName())
                                    == null) {
                                if (!lifeCycle.configManager.isModelMetricsAutoDetectEnabled()) {
                                    continue;
                                }

                                logger.info(
                                        "Registering auto detected backend metric: {}",
                                        parsedMetric);
                                this.metricCache.addAutoDetectMetricBackend(parsedMetric);
                            }

                            // Hostname is added as a dimension by default to backend metrics
                            List<String> dimensionValues = parsedMetric.getDimensionValues();
                            dimensionValues.add(parsedMetric.getHostName());

                            this.metricCache
                                    .getMetricBackend(parsedMetric.getMetricName())
                                    .addOrUpdate(
                                            dimensionValues,
                                            parsedMetric.getRequestId(),
                                            Double.parseDouble(parsedMetric.getValue()));
                        } catch (Exception e) {
                            logger.error(
                                    "Failed to update backend metric ",
                                    parsedMetric.getMetricName(),
                                    ": ",
                                    e);
                        }
                        continue;
                    }

                    matcher = WORKER_START_PATTERN.matcher(result);
                    if (matcher.matches()) {
                        lifeCycle.setSuccess(true);
                    } else {
                        matcher = WORKER_PID_PATTERN.matcher(result);
                        if (matcher.matches()) {
                            lifeCycle.setPid(Integer.parseInt(matcher.group(3)));
                        }
                    }
                    if (error) {
                        loggerModelOutput.warn(result);
                    } else {
                        loggerModelOutput.info(result);
                    }
                }
            } catch (Exception e) {
                logger.error("Couldn't create scanner - {}", getName(), e);
            } finally {
                logger.info("Stopped Scanner - {}", getName());
                lifeCycle.setSuccess(false);
                try {
                    is.close();
                } catch (IOException e) {
                    logger.error("Failed to close stream for thread {}", this.getName(), e);
                }
            }
        }
    }
}
