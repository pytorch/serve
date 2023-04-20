package org.pytorch.serve.metrics.configuration;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.composer.ComposerException;
import org.yaml.snakeyaml.constructor.Constructor;

public class MetricConfiguration {
    private static final Logger logger = LoggerFactory.getLogger(MetricConfiguration.class);
    private List<String> dimensions;

    @SuppressWarnings("checkstyle:MemberName")
    private MetricTypes ts_metrics;

    @SuppressWarnings("checkstyle:MemberName")
    private MetricTypes model_metrics;

    public void setDimensions(List<String> dimensions) {
        this.dimensions = dimensions;
    }

    public List<String> getDimensions() {
        return this.dimensions;
    }

    @SuppressWarnings("checkstyle:MethodName")
    public void setTs_metrics(MetricTypes tsMetrics) {
        this.ts_metrics = tsMetrics;
    }

    @SuppressWarnings("checkstyle:MethodName")
    public MetricTypes getTs_metrics() {
        return this.ts_metrics;
    }

    @SuppressWarnings("checkstyle:MethodName")
    public void setModel_metrics(MetricTypes modelMetrics) {
        // The Hostname dimension is included by default for model metrics
        modelMetrics.setCounter(this.addHostnameDimensionToMetrics(modelMetrics.getCounter()));
        modelMetrics.setGauge(this.addHostnameDimensionToMetrics(modelMetrics.getGauge()));
        modelMetrics.setHistogram(this.addHostnameDimensionToMetrics(modelMetrics.getHistogram()));
        this.model_metrics = modelMetrics;
    }

    @SuppressWarnings("checkstyle:MethodName")
    public MetricTypes getModel_metrics() {
        return this.model_metrics;
    }

    public void validate() {
        if (this.ts_metrics != null) {
            ts_metrics.validate();
        }

        if (this.model_metrics != null) {
            model_metrics.validate();
        }
    }

    public static MetricConfiguration loadConfiguration(String configFilePath)
            throws FileNotFoundException, ComposerException, RuntimeException {
        Constructor constructor = new Constructor(MetricConfiguration.class);
        Yaml yaml = new Yaml(constructor);
        FileInputStream inputStream = new FileInputStream(new File(configFilePath));
        MetricConfiguration config = yaml.load(inputStream);
        config.validate();
        logger.info("Successfully loaded metrics configuration from {}", configFilePath);

        return config;
    }

    private List<MetricSpecification> addHostnameDimensionToMetrics(
            List<MetricSpecification> metricsSpec) {
        if (metricsSpec == null) {
            return metricsSpec;
        }

        for (MetricSpecification spec : metricsSpec) {
            List<String> dimensions = spec.getDimensions();
            dimensions.add("Hostname");
            spec.setDimensions(dimensions);
        }

        return metricsSpec;
    }
}
