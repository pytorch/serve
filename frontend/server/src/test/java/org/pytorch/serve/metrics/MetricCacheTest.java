package org.pytorch.serve.metrics;

import java.io.FileNotFoundException;
import java.io.IOException;
import org.pytorch.serve.metrics.format.prometheous.PrometheusCounter;
import org.pytorch.serve.metrics.format.prometheous.PrometheusGauge;
import org.pytorch.serve.util.ConfigManager;
import org.testng.Assert;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

public class MetricCacheTest {
    @BeforeMethod
    public void setupConfigManager() throws IOException {
        ConfigManager.Arguments args = new ConfigManager.Arguments();
        ConfigManager.init(args);
    }

    @Test
    public void testMetricCacheLoadValidConfiguration() throws FileNotFoundException {
        ConfigManager.getInstance()
                .setProperty(
                        "metrics_config", "src/test/resources/metrics/valid_configuration.yaml");
        MetricCache.init();
        MetricCache metricCache = MetricCache.getInstance();
        Assert.assertEquals(
                metricCache.getMetricFrontend("Requests2XX").getClass(), PrometheusCounter.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("InferenceRequestsTotal").getClass(),
                PrometheusCounter.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("QueueTime").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("WorkerThreadTime").getClass(),
                PrometheusGauge.class);
        Assert.assertEquals(metricCache.getMetricFrontend("IvalidMetric"), null);
        Assert.assertEquals(
                metricCache.getMetricBackend("HandlerTime").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricBackend("PredictionTime").getClass(), PrometheusGauge.class);
        Assert.assertEquals(metricCache.getMetricBackend("IvalidMetric"), null);
    }
}
