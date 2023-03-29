package org.pytorch.serve.metrics;

import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.pytorch.serve.metrics.configuration.MetricConfiguration;
import org.pytorch.serve.metrics.configuration.MetricSpecification;
import org.pytorch.serve.util.ConfigManager;

public final class MetricCache {
    private static MetricCache instance;
    private Map<String, IMetric> metricsFrontend;
    private ConcurrentMap<String, IMetric> metricsBackend;

    private MetricCache() throws FileNotFoundException {
        this.metricsFrontend = new HashMap<String, IMetric>();
        this.metricsBackend = new ConcurrentHashMap<String, IMetric>();

        String metricsConfigPath = ConfigManager.getInstance().getMetricsConfigPath();
        MetricConfiguration config = MetricConfiguration.loadConfiguration(metricsConfigPath);

        MetricBuilder.MetricMode metricMode = MetricBuilder.MetricMode.LOG;
        String metricConfigMode = config.getMode();
        if (metricConfigMode != null && metricConfigMode.toLowerCase().contains("prometheus")) {
            metricMode = MetricBuilder.MetricMode.PROMETHEUS;
        }

        if (config.getTs_metrics() != null) {
            addMetrics(
                    config.getTs_metrics().getCounter(),
                    metricMode,
                    MetricBuilder.MetricContext.FRONTEND,
                    MetricBuilder.MetricType.COUNTER);
            addMetrics(
                    config.getTs_metrics().getGauge(),
                    metricMode,
                    MetricBuilder.MetricContext.FRONTEND,
                    MetricBuilder.MetricType.GAUGE);
            addMetrics(
                    config.getTs_metrics().getHistogram(),
                    metricMode,
                    MetricBuilder.MetricContext.FRONTEND,
                    MetricBuilder.MetricType.HISTOGRAM);
        }

        if (config.getModel_metrics() != null) {
            addMetrics(
                    config.getModel_metrics().getCounter(),
                    metricMode,
                    MetricBuilder.MetricContext.BACKEND,
                    MetricBuilder.MetricType.COUNTER);
            addMetrics(
                    config.getModel_metrics().getGauge(),
                    metricMode,
                    MetricBuilder.MetricContext.BACKEND,
                    MetricBuilder.MetricType.GAUGE);
            addMetrics(
                    config.getModel_metrics().getHistogram(),
                    metricMode,
                    MetricBuilder.MetricContext.BACKEND,
                    MetricBuilder.MetricType.HISTOGRAM);
        }
    }

    private void addMetricFrontend(String metricName, IMetric metric) {
        metricsFrontend.put(metricName, metric);
    }

    private void addMetricBackend(String metricName, IMetric metric) {
        metricsBackend.put(metricName, metric);
    }

    private void addMetrics(
            List<MetricSpecification> metricsSpec,
            MetricBuilder.MetricMode metricMode,
            MetricBuilder.MetricContext metricContext,
            MetricBuilder.MetricType metricType) {
        if (metricsSpec == null || metricsSpec.isEmpty()) {
            return;
        }

        for (MetricSpecification spec : metricsSpec) {
            if (metricContext == MetricBuilder.MetricContext.FRONTEND) {
                this.addMetricFrontend(
                        spec.getName(),
                        MetricBuilder.build(
                                metricMode,
                                metricContext,
                                metricType,
                                spec.getName(),
                                spec.getUnit(),
                                spec.getDimensions()));
            } else if (metricContext == MetricBuilder.MetricContext.BACKEND) {
                this.addMetricBackend(
                        spec.getName(),
                        MetricBuilder.build(
                                metricMode,
                                metricContext,
                                metricType,
                                spec.getName(),
                                spec.getUnit(),
                                spec.getDimensions()));
            }
        }
    }

    public static void init() throws FileNotFoundException {
        if (instance != null) {
            return;
        }

        instance = new MetricCache();
    }

    public static MetricCache getInstance() {
        return instance;
    }

    public IMetric getMetricFrontend(String metricName) {
        return metricsFrontend.get(metricName);
    }

    public IMetric getMetricBackend(String metricName) {
        return metricsBackend.get(metricName);
    }
}
