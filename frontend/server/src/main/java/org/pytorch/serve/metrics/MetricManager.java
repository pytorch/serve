package org.pytorch.serve.metrics;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.ModelManager;

public final class MetricManager {

    private static final MetricManager METRIC_MANAGER = new MetricManager();
    private List<Metric> metrics;

    private MetricManager() {
        metrics = Collections.emptyList();
    }

    public static MetricManager getInstance() {
        return METRIC_MANAGER;
    }

    public static void scheduleMetrics(ConfigManager configManager) {
        MetricCollector metricCollector = new MetricCollector(configManager);
        ModelManager.getInstance()
                .getScheduler()
                .scheduleAtFixedRate(
                        metricCollector,
                        0,
                        configManager.getMetricTimeInterval(),
                        TimeUnit.SECONDS);
    }

    public synchronized List<Metric> getMetrics() {
        return metrics;
    }

    public synchronized void setMetrics(List<Metric> metrics) {
        this.metrics = metrics;
    }
}
