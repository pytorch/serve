package org.pytorch.serve.metrics.configuration;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.List;
import org.pytorch.serve.util.ConfigManager;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.composer.ComposerException;
import org.yaml.snakeyaml.constructor.Constructor;

public class MetricConfiguration {
    private String mode;
    private List<String> dimensions;

    @SuppressWarnings("checkstyle:MemberName")
    private MetricTypes ts_metrics;

    @SuppressWarnings("checkstyle:MemberName")
    private MetricTypes model_metrics;

    public void setMode(String mode) {
        this.mode = mode;
    }

    public String getMode() {
        return mode;
    }

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

        String metricsMode = ConfigManager.getInstance().getMetricsMode();
        if (metricsMode != null) {
            config.setMode(metricsMode.toLowerCase());
        }

        return config;
    }
}
