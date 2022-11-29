package org.pytorch.serve.wlm;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.pytorch.serve.metrics.Metric;
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
    private String launcherArgs;
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

    public ArrayList<String> launcherArgsToList() {
        ArrayList<String> arrlist = new ArrayList<String>();
        arrlist.add("-m");
        arrlist.add("intel_extension_for_pytorch.cpu.launch");
        if (launcherArgs != null && launcherArgs.length() > 1) {
            String[] argarray = launcherArgs.split(" ");
            for (int i = 0; i < argarray.length; i++) {
                arrlist.add(argarray[i]);
            }
        }
        return arrlist;
    }

    public boolean isLauncherAvailable()
            throws WorkerInitializationException, InterruptedException {
        boolean launcherAvailable = false;
        try {
            ArrayList<String> cmd = new ArrayList<String>();
            cmd.add("python");
            ArrayList<String> args = launcherArgsToList();
            cmd.addAll(args);
            cmd.add("--no_python");
            // try launching dummy command to check launcher availability
            String dummyCmd = "hostname";
            cmd.add(dummyCmd);

            String[] cmdList = new String[cmd.size()];
            cmdList = cmd.toArray(cmdList);

            Process processLauncher = Runtime.getRuntime().exec(cmdList);
            int ret = processLauncher.waitFor();
            launcherAvailable = (ret == 0);
        } catch (IOException | InterruptedException e) {
            throw new WorkerInitializationException("Failed to start launcher", e);
        }
        return launcherAvailable;
    }

    public void startWorker(int port) throws WorkerInitializationException, InterruptedException {
        switch (model.getRuntimeType()) {
            case LSP:
                logger.info("LSP startWorker");
                startWorkerCPP(port, "LSP");
                break;
            case LDP:
                logger.info("LDP startWorker");
                startWorkerCPP(port, "LDP");
                break;
            default:
                startWorkerPython(port);
                break;
        }
    }

    private void startWorkerPython(int port)
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
        argl.add(EnvironmentUtils.getPythonRunTime(model));

        if (configManager.isCPULauncherEnabled()) {
            launcherArgs = configManager.getCPULauncherArgs();
            boolean launcherAvailable = isLauncherAvailable();
            if (launcherAvailable) {
                ArrayList<String> args = launcherArgsToList();
                argl.addAll(args);

                // multi-worker core pinning
                if (this.numWorker > 1) {
                    argl.add("--ninstances");
                    argl.add(String.valueOf(this.numWorker));
                    argl.add("--instance_idx");
                    // instance_idx is 0-indexed
                    argl.add(String.valueOf(this.currNumRunningWorkers));
                }

            } else {
                logger.warn(
                        "CPU launcher is enabled but launcher is not available. Proceeding without launcher.");
            }
        }

        argl.add(new File(workingDir, "ts/model_service_worker.py").getAbsolutePath());
        argl.add("--sock-type");
        argl.add(connector.getSocketType());
        argl.add(connector.isUds() ? "--sock-name" : "--port");
        argl.add(connector.getSocketPath());

        String[] envp =
                EnvironmentUtils.getEnvString(
                        workingDir.getAbsolutePath(),
                        modelPath.getAbsolutePath(),
                        model.getModelArchive().getManifest().getModel().getHandler());

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

    private void startWorkerCPP(int port, String runtimeType)
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
        argl.add("--logger_config_path");
        if (ConfigManager.getInstance().getTsCppLogConfig() != null) {
            argl.add(ConfigManager.getInstance().getTsCppLogConfig());
        } else {
            argl.add(configManager.getModelServerHome() + "/ts/cpp/resources/logging.config");
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

    public synchronized void terminateIOStreams() {
        if (errReader != null) {
            logger.warn("terminateIOStreams() threadName={}", errReader.getName());
            errReader.terminate();
        }
        if (outReader != null) {
            logger.warn("terminateIOStreams() threadName={}", outReader.getName());
            outReader.terminate();
        }
    }

    public synchronized void exit() {
        if (process != null) {
            process.destroyForcibly();
            connector.clean();
            terminateIOStreams();
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

        private InputStream is;
        private boolean error;
        private WorkerLifeCycle lifeCycle;
        private AtomicBoolean isRunning = new AtomicBoolean(true);
        private static final Logger loggerModelMetrics =
                LoggerFactory.getLogger(ConfigManager.MODEL_METRICS_LOGGER);
        private static final Logger loggerModelOutput =
                LoggerFactory.getLogger(ConfigManager.MODEL_LOGGER);

        public ReaderThread(String name, InputStream is, boolean error, WorkerLifeCycle lifeCycle) {
            super(name + (error ? "-stderr" : "-stdout"));
            this.is = is;
            this.error = error;
            this.lifeCycle = lifeCycle;
        }

        public void terminate() {
            isRunning.set(false);
        }

        @Override
        public void run() {
            try (Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name())) {
                while (isRunning.get() && scanner.hasNext()) {
                    String result = scanner.nextLine();
                    if (result == null) {
                        break;
                    }
                    int metricLogStartSubstringIndex = result.indexOf(METRIC_LOG_START_SUBSTRING);
                    if (metricLogStartSubstringIndex >= 0) {
                        Metric parsedMetric =
                                Metric.parse(
                                        result.substring(
                                                metricLogStartSubstringIndex
                                                        + METRIC_LOG_START_SUBSTRING.length()));
                        if (parsedMetric != null) {
                            loggerModelMetrics.info(parsedMetric.toString());
                        } else {
                            logger.error("Failed to parse metrics line: \"{}\".", result);
                        }
                        continue;
                    }

                    Matcher matcher = PID_LOG_PATTERN.matcher(result);
                    if (result.endsWith("Torch worker started.")) {
                        lifeCycle.setSuccess(true);
                    } else if (matcher.matches()) {
                        lifeCycle.setPid(Integer.parseInt(matcher.group(1)));
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
