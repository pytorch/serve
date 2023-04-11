package org.pytorch.serve.metrics.configuration;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import org.pytorch.serve.util.ConfigManager;
import org.testng.Assert;
import org.testng.annotations.Test;
import org.yaml.snakeyaml.composer.ComposerException;

public class MetricConfigurationTest {
    @Test
    public void testLoadValidConfiguration()
            throws FileNotFoundException, ComposerException, RuntimeException {
        MetricConfiguration config =
                MetricConfiguration.loadConfiguration(
                        "src/test/resources/metrics/valid_configuration.yaml");

        Assert.assertEquals(config.getMode(), "log");

        Assert.assertEquals(
                config.getDimensions(),
                new ArrayList<String>(
                        Arrays.asList("ModelName", "ModelVersion", "Level", "Hostname")));

        Assert.assertEquals(config.getTs_metrics().getCounter().size(), 2);

        MetricSpecification spec = config.getTs_metrics().getCounter().get(0);
        Assert.assertEquals(spec.getName(), "Requests2XX");
        Assert.assertEquals(spec.getUnit(), "Count");
        Assert.assertEquals(
                spec.getDimensions(), new ArrayList<String>(Arrays.asList("Level", "Hostname")));

        spec = config.getTs_metrics().getCounter().get(1);
        Assert.assertEquals(spec.getName(), "InferenceRequestsTotal");
        Assert.assertEquals(spec.getUnit(), "Count");
        Assert.assertEquals(
                spec.getDimensions(),
                new ArrayList<String>(Arrays.asList("ModelName", "ModelVersion", "Hostname")));

        Assert.assertEquals(config.getTs_metrics().getGauge().size(), 2);

        spec = config.getTs_metrics().getGauge().get(0);
        Assert.assertEquals(spec.getName(), "QueueTime");
        Assert.assertEquals(spec.getUnit(), "Milliseconds");
        Assert.assertEquals(
                spec.getDimensions(), new ArrayList<String>(Arrays.asList("Level", "Hostname")));

        spec = config.getTs_metrics().getGauge().get(1);
        Assert.assertEquals(spec.getName(), "WorkerThreadTime");
        Assert.assertEquals(spec.getUnit(), "Milliseconds");
        Assert.assertEquals(
                spec.getDimensions(), new ArrayList<String>(Arrays.asList("Level", "Hostname")));

        Assert.assertEquals(config.getTs_metrics().getHistogram(), null);

        Assert.assertEquals(config.getModel_metrics().getCounter(), null);

        Assert.assertEquals(config.getModel_metrics().getGauge().size(), 2);

        spec = config.getModel_metrics().getGauge().get(0);
        Assert.assertEquals(spec.getName(), "HandlerTime");
        Assert.assertEquals(spec.getUnit(), "ms");
        Assert.assertEquals(
                spec.getDimensions(), new ArrayList<String>(Arrays.asList("ModelName", "Level")));

        spec = config.getModel_metrics().getGauge().get(1);
        Assert.assertEquals(spec.getName(), "PredictionTime");
        Assert.assertEquals(spec.getUnit(), "ms");
        Assert.assertEquals(
                spec.getDimensions(), new ArrayList<String>(Arrays.asList("ModelName", "Level")));

        Assert.assertEquals(config.getModel_metrics().getHistogram(), null);
    }

    @Test
    public void testLoadValidConfigurationEmptyMetricDimensions()
            throws FileNotFoundException, ComposerException, RuntimeException {
        MetricConfiguration config =
                MetricConfiguration.loadConfiguration(
                        "src/test/resources/metrics/valid_configuration_empty_metric_dimensions.yaml");

        Assert.assertEquals(config.getMode(), "log");

        Assert.assertEquals(config.getDimensions(), null);

        Assert.assertEquals(config.getTs_metrics().getCounter().size(), 1);

        MetricSpecification spec = config.getTs_metrics().getCounter().get(0);
        Assert.assertEquals(spec.getName(), "InferenceRequestsTotal");
        Assert.assertEquals(spec.getUnit(), "Count");
        Assert.assertEquals(spec.getDimensions(), null);

        Assert.assertEquals(config.getTs_metrics().getGauge(), null);

        Assert.assertEquals(config.getTs_metrics().getHistogram(), null);

        Assert.assertEquals(config.getModel_metrics(), null);
    }

    @Test
    public void testLoadInvalidConfigurationMissingDimension() {
        Assert.assertThrows(
                ComposerException.class,
                () ->
                        MetricConfiguration.loadConfiguration(
                                "src/test/resources/metrics/invalid_configuration_missing_dimension.yaml"));
    }

    @Test
    public void testLoadInvalidConfigurationMissingMetricName() {
        Assert.assertThrows(
                RuntimeException.class,
                () ->
                        MetricConfiguration.loadConfiguration(
                                "src/test/resources/metrics/invalid_configuration_missing_metric_name.yaml"));
    }

    @Test
    public void testLoadInvalidConfigurationMissingMetricUnit() {
        Assert.assertThrows(
                RuntimeException.class,
                () ->
                        MetricConfiguration.loadConfiguration(
                                "src/test/resources/metrics/invalid_configuration_missing_metric_unit.yaml"));
    }

    @Test
    public void testLoadValidConfigurationModeEnvironmentVariable()
            throws IOException, FileNotFoundException {
        ConfigManager.getInstance().setProperty("metrics_mode", "test_metrics_mode");
        MetricConfiguration config =
                MetricConfiguration.loadConfiguration(
                        "src/test/resources/metrics/valid_configuration.yaml");
        Assert.assertEquals(config.getMode(), "test_metrics_mode");
    }
}
