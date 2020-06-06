package org.pytorch.serve.wlm;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;
import org.pytorch.serve.archive.Manifest;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class WorkerLifeCycle {

    private static final Logger logger = LoggerFactory.getLogger(WorkerLifeCycle.class);

    private ConfigManager configManager;
    private Model model;
    private int pid = -1;
    private Process process;
    private CountDownLatch latch;
    private boolean success;
    private Connector connector;
    private ReaderThread errReader;
    private ReaderThread outReader;

    public WorkerLifeCycle(ConfigManager configManager, Model model) {
        this.configManager = configManager;
        this.model = model;
    }

    public Process getProcess() {
        return process;
    }

    private String[] getEnvString(String cwd, String modelPath, String handler) {
        ArrayList<String> envList = new ArrayList<>();
        Pattern blackList = configManager.getBlacklistPattern();
        StringBuilder pythonPath = new StringBuilder();

        if (handler != null && handler.contains(":")) {
            String handlerFile = handler;
            handlerFile = handler.split(":")[0];
            if (handlerFile.contains("/")) {
                handlerFile = handlerFile.substring(0, handlerFile.lastIndexOf('/'));
            }

            pythonPath.append(handlerFile).append(File.pathSeparatorChar);
        }

        HashMap<String, String> environment = new HashMap<>(System.getenv());
        environment.putAll(configManager.getBackendConfiguration());

        if (System.getenv("PYTHONPATH") != null) {
            pythonPath.append(System.getenv("PYTHONPATH")).append(File.pathSeparatorChar);
        }

        pythonPath.append(modelPath);

        if (!cwd.contains("site-packages") && !cwd.contains("dist-packages")) {
            pythonPath.append(File.pathSeparatorChar).append(cwd);
        }

        environment.put("PYTHONPATH", pythonPath.toString());

        for (Map.Entry<String, String> entry : environment.entrySet()) {
            if (!blackList.matcher(entry.getKey()).matches()) {
                envList.add(entry.getKey() + '=' + entry.getValue());
            }
        }

        return envList.toArray(new String[0]); // NOPMD
    }

    public void startWorker(int port) throws WorkerInitializationException, InterruptedException {
        File workingDir = new File(configManager.getModelServerHome());
        File modelPath;
        setPort(port);
        try {
            modelPath = model.getModelDir().getCanonicalFile();
        } catch (IOException e) {
            throw new WorkerInitializationException("Failed get TS home directory", e);
        }

        String[] args = new String[6];
        Manifest.RuntimeType runtime = model.getModelArchive().getManifest().getRuntime();
        if (runtime == Manifest.RuntimeType.PYTHON) {
            args[0] = configManager.getPythonExecutable();
        } else {
            args[0] = runtime.getValue();
        }
        args[1] = new File(workingDir, "ts/model_service_worker.py").getAbsolutePath();
        args[2] = "--sock-type";
        args[3] = connector.getSocketType();
        args[4] = connector.isUds() ? "--sock-name" : "--port";
        args[5] = connector.getSocketPath();

        String[] envp =
                getEnvString(
                        workingDir.getAbsolutePath(),
                        modelPath.getAbsolutePath(),
                        model.getModelArchive().getManifest().getModel().getHandler());

        try {
            latch = new CountDownLatch(1);

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
        private static final org.apache.log4j.Logger loggerModelMetrics =
                org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_METRICS_LOGGER);

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
                    if (result.startsWith("[METRICS]")) {
                        loggerModelMetrics.info(Metric.parse(result.substring(9)));
                        continue;
                    }

                    if ("Torch worker started.".equals(result)) {
                        lifeCycle.setSuccess(true);
                    } else if (result.startsWith("[PID]")) {
                        lifeCycle.setPid(Integer.parseInt(result.substring("[PID]".length())));
                    }
                    if (error) {
                        logger.warn(result);
                    } else {
                        logger.info(result);
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
