package org.pytorch.serve.metrics;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.io.IOUtils;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.messages.EnvironmentUtils;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkerThread;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MetricCollector implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(MetricCollector.class);
    private static final org.apache.log4j.Logger loggerMetrics =
            org.apache.log4j.Logger.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);
    private ConfigManager configManager;

    public MetricCollector(ConfigManager configManager) {
        this.configManager = configManager;
    }

    @Override
    public void run() {
        try {
            // Collect System level Metrics
            String[] args = new String[2];
            args[0] = configManager.getPythonExecutable();
            args[1] = "ts/metrics/metric_collector.py";
            File workingDir = new File(configManager.getModelServerHome());

            String[] envp = EnvironmentUtils.getEnvString(workingDir.getAbsolutePath(), null, null);

            final Process p = Runtime.getRuntime().exec(args, envp, workingDir); // NOPMD
            ModelManager modelManager = ModelManager.getInstance();
            Map<Integer, WorkerThread> workerMap = modelManager.getWorkers();
            try (OutputStream os = p.getOutputStream()) {
                writeWorkerPids(workerMap, os);
            }

            new Thread(
                            () -> {
                                try {
                                    String error =
                                            IOUtils.toString(
                                                    p.getErrorStream(), StandardCharsets.UTF_8);
                                    if (!error.isEmpty()) {
                                        logger.error(error);
                                    }
                                } catch (IOException e) {
                                    logger.error("", e);
                                }
                            })
                    .start();

            MetricManager metricManager = MetricManager.getInstance();
            try (BufferedReader reader =
                    new BufferedReader(
                            new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                List<Metric> metricsSystem = new ArrayList<>();
                metricManager.setMetrics(metricsSystem);

                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.isEmpty()) {
                        break;
                    }
                    Metric metric = Metric.parse(line);
                    if (metric == null) {
                        logger.warn("Parse metrics failed: " + line);
                    } else {
                        loggerMetrics.info(metric);
                        metricsSystem.add(metric);
                    }
                }

                // Collect process level metrics
                while ((line = reader.readLine()) != null) {
                    String[] tokens = line.split(":");
                    if (tokens.length != 2) {
                        continue;
                    }

                    Integer pid = Integer.valueOf(tokens[0]);
                    WorkerThread worker = workerMap.get(pid);
                    worker.setMemory(Long.parseLong(tokens[1]));
                }
            }
        } catch (IOException e) {
            logger.error("", e);
        }
    }

    private void writeWorkerPids(Map<Integer, WorkerThread> workerMap, OutputStream os)
            throws IOException {
        boolean first = true;
        for (Integer pid : workerMap.keySet()) {
            if (pid < 0) {
                logger.warn("worker pid is not available yet.");
                continue;
            }
            if (first) {
                first = false;
            } else {
                IOUtils.write(",", os, StandardCharsets.UTF_8);
            }
            IOUtils.write(pid.toString(), os, StandardCharsets.UTF_8);
        }
        os.write('\n');
    }
}
