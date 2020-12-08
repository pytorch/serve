package org.pytorch.serve.wlm;

import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.pytorch.serve.util.SharedNamedPipeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class WorkerLifeCycle {

    private static final Logger logger = LoggerFactory.getLogger(WorkerLifeCycle.class);

    private ConfigManager configManager;
    private Model model;
    private int pid = -1;
    private CountDownLatch latch;
    private boolean success;
    private Connector connector;
    private ReaderThread errReader;
    private ReaderThread outReader;
    private int port;

    private static final int STD_OUT_POLL_INTERVAL = 1000;
    private static final int STD_OUT_POLL_ATTEMPTS = 10;


    public WorkerLifeCycle(ConfigManager configManager, Model model, int port) {
        this.configManager = configManager;
        this.model = model;
        this.connector = new Connector(port);
        this.port = port;
    }

    public static void awaitFileCreate(String path)
            throws WorkerInitializationException {
        int retry = STD_OUT_POLL_ATTEMPTS;
        while (!(new File(path).exists())) {
            if (--retry <= 0) {
                throw new WorkerInitializationException("Worker std out file was not created in time");
            }
            try {
                Thread.sleep(STD_OUT_POLL_INTERVAL);
            } catch (InterruptedException e) {
                logger.info("Waiting to startup");
            }
        }
    }

    public void startWorker(int port) throws WorkerInitializationException, InterruptedException {

        try {
            latch = new CountDownLatch(1);

            synchronized (this) {

                String threadName =
                        "W-" + port + '-' + model.getModelVersionName().getVersionedModelName();


                SharedNamedPipeUtils.cleanupSharedNamedPipePathFiles(Integer.toString(port));

                String stdOutFile = SharedNamedPipeUtils.getSharedNamedPipeStdOut(Integer.toString(port));
                String stdErrFile = SharedNamedPipeUtils.getSharedNamedPipeStdErr(Integer.toString(port));

                awaitFileCreate(stdOutFile);

                logger.info("StdOut file created - " + stdOutFile);


                errReader = new ReaderThread(threadName, new File(stdOutFile), true, this);
                outReader = new ReaderThread(threadName, new File(stdErrFile), false, this);

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
        } catch (Exception e) {
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
        terminateIOStreams();
    }

    public synchronized Integer getExitValue() {
        return null;
    }

    public void setSuccess(boolean success) {
        this.success = success;
        latch.countDown();
    }

    public synchronized int getPid() {
        return pid;
    }

    public synchronized int getPort() {
        return port;
    }

    public synchronized void setPid(int pid) {
        this.pid = pid;
    }

    private synchronized void setPort(int port) {
        connector = new Connector(port);
    }

    private static final class ReaderThread extends Thread {

        private static final org.apache.log4j.Logger loggerModelMetrics =
                org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_METRICS_LOGGER);
        private final int POLL_FREQUENCY = 500;
        private File file;
        private long lastReadPosition;
        private boolean error;
        private WorkerLifeCycle lifeCycle;
        private AtomicBoolean isRunning = new AtomicBoolean(true);

        public ReaderThread(String name, File file, boolean error, WorkerLifeCycle lifeCycle) {
            super(name + (error ? "-stderr" : "-stdout"));
            this.file = file;
            this.error = error;
            this.lifeCycle = lifeCycle;
            this.lastReadPosition = 0;
        }

        public void terminate() {
            isRunning.set(false);
        }


        @Override
        public void run() {

            while (isRunning.get()) {

                try {
                    Thread.sleep(POLL_FREQUENCY);
                } catch (InterruptedException e) {
                }

                long fileLength = file.length();
                if (fileLength > lastReadPosition) {

                    try {

                        RandomAccessFile readWriteFileAccess = new RandomAccessFile(file, "rw");
                        String result = null;
                        while ((result = readWriteFileAccess.readLine()) != null) {

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
                        lastReadPosition = readWriteFileAccess.getFilePointer();
                        readWriteFileAccess.close();
                    } catch (IOException e) {
                        logger.error("Error while reading file - " + file.getName());
                    } finally {
                        logger.info("Stopped Scanner - {}", getName());
                        lifeCycle.setSuccess(false);
                    }
                }
            }
        }
    }
}
