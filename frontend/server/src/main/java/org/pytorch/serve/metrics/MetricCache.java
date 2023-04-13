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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class MetricCache {
    private static final Logger logger = LoggerFactory.getLogger(MetricCache.class);
    private static MetricCache instance;
    private MetricConfiguration config;
    private Map<String, IMetric> metricsFrontend;
    private ConcurrentMap<String, IMetric> metricsBackend;

    private MetricCache() throws FileNotFoundException {
        this.metricsFrontend = new HashMap<String, IMetric>();
        this.metricsBackend = new ConcurrentHashMap<String, IMetric>();

        String metricsConfigPath = ConfigManager.getInstance().getMetricsConfigPath();
        try {
            this.config = MetricConfiguration.loadConfiguration(metricsConfigPath);
        } catch (FileNotFoundException | RuntimeException e) {
            logger.error("Failed to initialize metrics cache: ", e);
            return;
        }

        MetricBuilder.MetricMode metricsMode = MetricBuilder.MetricMode.LOG;
        String metricsConfigMode = ConfigManager.getInstance().getMetricsMode();
        if (metricsConfigMode != null && metricsConfigMode.toLowerCase().contains("prometheus")) {
            metricsMode = MetricBuilder.MetricMode.PROMETHEUS;
        }

        if (this.config.getTs_metrics() != null) {
            addMetricsFrontend(
                    this.config.getTs_metrics().getCounter(),
                    metricsMode,
                    MetricBuilder.MetricType.COUNTER);
            addMetricsFrontend(
                    this.config.getTs_metrics().getGauge(),
                    metricsMode,
                    MetricBuilder.MetricType.GAUGE);
            addMetricsFrontend(
                    this.config.getTs_metrics().getHistogram(),
                    metricsMode,
                    MetricBuilder.MetricType.HISTOGRAM);
        }

        if (this.config.getModel_metrics() != null) {
            addMetricsBackend(
                    this.config.getModel_metrics().getCounter(),
                    metricsMode,
                    MetricBuilder.MetricType.COUNTER);
            addMetricsBackend(
                    this.config.getModel_metrics().getGauge(),
                    metricsMode,
                    MetricBuilder.MetricType.GAUGE);
            addMetricsBackend(
                    this.config.getModel_metrics().getHistogram(),
                    metricsMode,
                    MetricBuilder.MetricType.HISTOGRAM);
        }
    }

    private void addMetricsFrontend(
            List<MetricSpecification> metricsSpec,
            MetricBuilder.MetricMode metricsMode,
            MetricBuilder.MetricType metricType) {
        if (metricsSpec == null) {
            return;
        }

        for (MetricSpecification spec : metricsSpec) {
            this.metricsFrontend.put(
                    spec.getName(),
                    MetricBuilder.build(
                            metricsMode,
                            metricType,
                            spec.getName(),
                            spec.getUnit(),
                            spec.getDimensions()));
        }
    }

    private void addMetricsBackend(
            List<MetricSpecification> metricsSpec,
            MetricBuilder.MetricMode metricsMode,
            MetricBuilder.MetricType metricType) {
        if (metricsSpec == null) {
            return;
        }

        for (MetricSpecification spec : metricsSpec) {
            this.metricsBackend.put(
                    spec.getName(),
                    MetricBuilder.build(
                            metricsMode,
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
