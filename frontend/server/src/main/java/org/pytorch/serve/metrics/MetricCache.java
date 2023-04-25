package org.pytorch.serve.metrics;

import java.io.FileNotFoundException;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.pytorch.serve.metrics.configuration.MetricConfiguration;
import org.pytorch.serve.metrics.configuration.MetricSpecification;
import org.pytorch.serve.util.ConfigManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class MetricCache {
    private static final Logger logger = LoggerFactory.getLogger(MetricCache.class);
    private static MetricCache instance;
    private MetricConfiguration config;
    private ConcurrentMap<String, IMetric> metricsFrontend;
    private ConcurrentMap<String, IMetric> metricsBackend;

    private MetricCache() throws FileNotFoundException {
        this.metricsFrontend = new ConcurrentHashMap<String, IMetric>();
        this.metricsBackend = new ConcurrentHashMap<String, IMetric>();

        String metricsConfigPath = ConfigManager.getInstance().getMetricsConfigPath();
        try {
            this.config = MetricConfiguration.loadConfiguration(metricsConfigPath);
        } catch (FileNotFoundException | RuntimeException e) {
            logger.error("Failed to load metrics configuration: ", e);
            return;
        }

        MetricBuilder.MetricMode metricsMode = MetricBuilder.MetricMode.LOG;
        String metricsConfigMode = ConfigManager.getInstance().getMetricsMode();
        if (metricsConfigMode != null && metricsConfigMode.toLowerCase().contains("prometheus")) {
            metricsMode = MetricBuilder.MetricMode.PROMETHEUS;
        }

        if (this.config.getTs_metrics() != null) {
            addMetrics(
                    this.metricsFrontend,
                    this.config.getTs_metrics().getCounter(),
                    metricsMode,
                    MetricBuilder.MetricType.COUNTER);
            addMetrics(
                    this.metricsFrontend,
                    this.config.getTs_metrics().getGauge(),
                    metricsMode,
                    MetricBuilder.MetricType.GAUGE);
            addMetrics(
                    this.metricsFrontend,
                    this.config.getTs_metrics().getHistogram(),
                    metricsMode,
                    MetricBuilder.MetricType.HISTOGRAM);
        }

        if (this.config.getModel_metrics() != null) {
            addMetrics(
                    this.metricsBackend,
                    this.config.getModel_metrics().getCounter(),
                    metricsMode,
                    MetricBuilder.MetricType.COUNTER);
            addMetrics(
                    this.metricsBackend,
                    this.config.getModel_metrics().getGauge(),
                    metricsMode,
                    MetricBuilder.MetricType.GAUGE);
            addMetrics(
                    this.metricsBackend,
                    this.config.getModel_metrics().getHistogram(),
                    metricsMode,
                    MetricBuilder.MetricType.HISTOGRAM);
        }
    }

    private void addMetrics(
            ConcurrentMap<String, IMetric> metricCache,
            List<MetricSpecification> metricsSpec,
            MetricBuilder.MetricMode metricMode,
            MetricBuilder.MetricType metricType) {
        if (metricsSpec == null) {
            return;
        }

        for (MetricSpecification spec : metricsSpec) {
            metricCache.put(
                    spec.getName(),
                    MetricBuilder.build(
                            metricMode,
                            metricType,
                            spec.getName(),
                            spec.getUnit(),
                            spec.getDimensions()));
        }
    }

    public static void init() throws FileNotFoundException {
        if (instance != null) {
            logger.error("Skip initializing metrics cache since it has already been initialized");
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
