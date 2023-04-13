package org.pytorch.serve.metrics;

import java.io.FileNotFoundException;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MetricCacheTest {
    @Test
    public void testMetricCacheLoadValidConfiguration() throws FileNotFoundException {
        MetricCache metricCache = MetricCache.getInstance();
        Assert.assertEquals(
                metricCache.getMetricFrontend("Requests2XX").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("Requests4XX").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("Requests5XX").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("ts_inference_requests_total").getClass(),
                LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("ts_inference_latency_microseconds").getClass(),
                LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("ts_queue_latency_microseconds").getClass(),
                LogMetric.class);
        Assert.assertEquals(metricCache.getMetricFrontend("QueueTime").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("WorkerThreadTime").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("WorkerLoadTime").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("CPUUtilization").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("MemoryUsed").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("MemoryAvailable").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("MemoryUtilization").getClass(), LogMetric.class);
        Assert.assertEquals(metricCache.getMetricFrontend("DiskUsage").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("DiskUtilization").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("DiskAvailable").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("GPUMemoryUtilization").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("GPUMemoryUsed").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricFrontend("GPUUtilization").getClass(), LogMetric.class);
        Assert.assertEquals(metricCache.getMetricFrontend("IvalidMetric"), null);
        Assert.assertEquals(
                metricCache.getMetricBackend("HandlerTime").getClass(), LogMetric.class);
        Assert.assertEquals(
                metricCache.getMetricBackend("PredictionTime").getClass(), LogMetric.class);
        Assert.assertEquals(metricCache.getMetricBackend("IvalidMetric"), null);
    }
}
