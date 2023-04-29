package org.pytorch.serve.metrics;

import org.pytorch.serve.metrics.format.prometheous.PrometheusCounter;
import org.pytorch.serve.metrics.format.prometheous.PrometheusGauge;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MetricCacheTest {
    @Test
    public void testMetricCacheLoadValidConfiguration() {
        MetricCache metricCache = MetricCache.getInstance();
        Assert.assertEquals(
                metricCache.getMetricFrontend("Requests2XX").getClass(), PrometheusCounter.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("Requests4XX").getClass(), PrometheusCounter.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("Requests5XX").getClass(), PrometheusCounter.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("ts_inference_requests_total").getClass(),
                PrometheusCounter.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("ts_inference_latency_microseconds").getClass(),
                PrometheusCounter.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("ts_queue_latency_microseconds").getClass(),
                PrometheusCounter.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("QueueTime").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("WorkerThreadTime").getClass(),
                PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("WorkerLoadTime").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("CPUUtilization").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("MemoryUsed").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("MemoryAvailable").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("MemoryUtilization").getClass(),
                PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("DiskUsage").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("DiskUtilization").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("DiskAvailable").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("GPUMemoryUtilization").getClass(),
                PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("GPUMemoryUsed").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("GPUUtilization").getClass(), PrometheusGauge.class);
        Assert.assertEquals(metricCache.getMetricFrontend("InvalidMetric"), null);
        Assert.assertEquals(
                metricCache.getMetricBackend("HandlerTime").getClass(), PrometheusGauge.class);
        Assert.assertEquals(
                metricCache.getMetricBackend("PredictionTime").getClass(), PrometheusGauge.class);
        Assert.assertEquals(metricCache.getMetricBackend("InvalidMetric"), null);
    }
}
