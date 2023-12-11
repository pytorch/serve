package org.pytorch.serve.metrics;

import io.prometheus.client.CollectorRegistry;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.Logger;
import org.apache.logging.log4j.core.appender.WriterAppender;
import org.pytorch.serve.metrics.format.prometheous.PrometheusCounter;
import org.pytorch.serve.metrics.format.prometheous.PrometheusGauge;
import org.pytorch.serve.metrics.format.prometheous.PrometheusHistogram;
import org.pytorch.serve.util.ConfigManager;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

@Test(singleThreaded = true)
public class MetricTest {
    private final String testMetricName = "TestMetric";
    private final String testMetricUnit = "ms";
    private final ArrayList<String> testMetricDimensionNames =
            new ArrayList<String>(Arrays.asList("ModelName", "Level", "Hostname"));
    private final ArrayList<String> testMetricDimensionValues =
            new ArrayList<String>(Arrays.asList("TestModel", "Model", "TestHost"));
    private final String testRequestId = "fa8639a8-d3fa-4a25-a80f-24463863fe0f";
    private final Logger loggerModelMetrics =
            (org.apache.logging.log4j.core.Logger)
                    LogManager.getLogger(ConfigManager.MODEL_METRICS_LOGGER);
    private final Logger loggerTsMetrics =
            (org.apache.logging.log4j.core.Logger)
                    LogManager.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);
    private final String modelMetricsAppenderName = "ModelMetricsAppender";
    private final String tsMetricsAppenderName = "TsMetricsAppender";
    private final StringWriter modelMetricsContent = new StringWriter();
    private final StringWriter tsMetricsContent = new StringWriter();
    private final WriterAppender modelMetricsAppender =
            WriterAppender.createAppender(
                    null, null, modelMetricsContent, modelMetricsAppenderName, true, false);
    private final WriterAppender tsMetricsAppender =
            WriterAppender.createAppender(
                    null, null, tsMetricsContent, tsMetricsAppenderName, true, false);

    @BeforeClass
    public void registerMetricLogAppenders() {
        loggerModelMetrics.addAppender(modelMetricsAppender);
        modelMetricsAppender.start();
        loggerTsMetrics.addAppender(tsMetricsAppender);
        tsMetricsAppender.start();
    }

    @BeforeMethod
    public void flushLogWriterStreams() {
        modelMetricsContent.flush();
        tsMetricsContent.flush();
    }

    @BeforeMethod
    public void clearPrometheusRegistry() {
        CollectorRegistry.defaultRegistry.clear();
    }

    @AfterClass
    public void unregisterMetricLogAppenders() {
        modelMetricsAppender.stop();
        loggerModelMetrics.removeAppender(modelMetricsAppender);
        tsMetricsAppender.stop();
        loggerTsMetrics.removeAppender(tsMetricsAppender);
    }

    @Test
    public void testBackendLogMetric() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.LOG,
                        MetricBuilder.MetricType.COUNTER,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), LogMetric.class);

        testMetric.addOrUpdate(testMetricDimensionValues, testRequestId, 1.0);
        String expectedMetricString =
                "TestMetric.ms:1.0|#ModelName:TestModel,Level:Model|#hostname:TestHost,"
                        + "requestID:fa8639a8-d3fa-4a25-a80f-24463863fe0f,timestamp:";
        Assert.assertTrue(modelMetricsContent.toString().contains(expectedMetricString));
    }

    @Test
    public void testFrontendLogMetric() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.LOG,
                        MetricBuilder.MetricType.GAUGE,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), LogMetric.class);

        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        String expectedMetricString =
                "TestMetric.ms:1.0|#ModelName:TestModel,Level:Model|#hostname:TestHost,timestamp:";
        Assert.assertTrue(tsMetricsContent.toString().contains(expectedMetricString));
    }

    @Test
    public void testBackendPrometheusCounter() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricType.COUNTER,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusCounter.class);

        testMetric.addOrUpdate(testMetricDimensionValues, testRequestId, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName,
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, testRequestId, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName,
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(3.0));
    }

    @Test
    public void testFrontendPrometheusCounter() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricType.COUNTER,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusCounter.class);

        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName,
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName,
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(3.0));
    }

    @Test
    public void testBackendPrometheusGauge() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricType.GAUGE,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusGauge.class);

        testMetric.addOrUpdate(testMetricDimensionValues, testRequestId, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName,
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, testRequestId, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName,
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(2.0));
    }

    @Test
    public void testFrontendPrometheusGauge() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricType.GAUGE,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusGauge.class);

        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName,
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName,
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(2.0));
    }

    @Test
    public void testBackendPrometheusHistogram() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricType.HISTOGRAM,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusHistogram.class);

        testMetric.addOrUpdate(testMetricDimensionValues, testRequestId, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName + "_sum",
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, testRequestId, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName + "_sum",
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(3.0));
    }

    @Test
    public void testFrontendPrometheusHistogram() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricType.HISTOGRAM,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusHistogram.class);

        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName + "_sum",
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName + "_sum",
                        testMetricDimensionNames.toArray(new String[0]),
                        testMetricDimensionValues.toArray(new String[0]));
        Assert.assertEquals(metricValue, Double.valueOf(3.0));
    }

    @Test
    public void testLegacyMetricMode() {
        ArrayList<String> legacyPrometheusMetrics =
                new ArrayList<String>(
                        Arrays.asList(
                                "ts_inference_requests_total",
                                "ts_inference_latency_microseconds",
                                "ts_queue_latency_microseconds"));

        for (String metricName : legacyPrometheusMetrics) {
            IMetric testMetric =
                    MetricBuilder.build(
                            MetricBuilder.MetricMode.LEGACY,
                            MetricBuilder.MetricType.COUNTER,
                            metricName,
                            testMetricUnit,
                            testMetricDimensionNames);
            Assert.assertEquals(testMetric.getClass(), PrometheusCounter.class);

            testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
            Double metricValue =
                    CollectorRegistry.defaultRegistry.getSampleValue(
                            metricName,
                            testMetricDimensionNames.toArray(new String[0]),
                            testMetricDimensionValues.toArray(new String[0]));
            Assert.assertEquals(metricValue, Double.valueOf(1.0));
            testMetric.addOrUpdate(testMetricDimensionValues, 2.0);
            metricValue =
                    CollectorRegistry.defaultRegistry.getSampleValue(
                            metricName,
                            testMetricDimensionNames.toArray(new String[0]),
                            testMetricDimensionValues.toArray(new String[0]));
            Assert.assertEquals(metricValue, Double.valueOf(3.0));
        }

        // Frontend log metric
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.LEGACY,
                        MetricBuilder.MetricType.GAUGE,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), LogMetric.class);

        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        String expectedMetricString =
                "TestMetric.ms:1.0|#ModelName:TestModel,Level:Model|#hostname:TestHost,timestamp:";
        Assert.assertTrue(tsMetricsContent.toString().contains(expectedMetricString));

        // Backend log metric
        testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.LEGACY,
                        MetricBuilder.MetricType.COUNTER,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), LogMetric.class);

        testMetric.addOrUpdate(testMetricDimensionValues, testRequestId, 1.0);
        expectedMetricString =
                "TestMetric.ms:1.0|#ModelName:TestModel,Level:Model|#hostname:TestHost,"
                        + "requestID:fa8639a8-d3fa-4a25-a80f-24463863fe0f,timestamp:";
        Assert.assertTrue(modelMetricsContent.toString().contains(expectedMetricString));
    }

    @Test
    public void testParseBackendMetricLogWithoutType() {
        String backendMetricLog =
                "HandlerTime.Milliseconds:71.77|#ModelName:mnist,Level:Model|#hostname:test-host,1699061430,6d1726a7-172c-4010-b671-01d71bacc451";
        Metric parsedMetric = Metric.parse(backendMetricLog);

        Assert.assertEquals("HandlerTime", parsedMetric.getMetricName());
        Assert.assertEquals("Milliseconds", parsedMetric.getUnit());
        Assert.assertEquals("71.77", parsedMetric.getValue());
        List<String> dimensionNames = new ArrayList<String>();
        for (Dimension dimension : parsedMetric.getDimensions()) {
            dimensionNames.add(dimension.getName());
        }
        Assert.assertEquals(Arrays.asList("ModelName", "Level"), dimensionNames);
        List<String> dimensionValues = new ArrayList<String>();
        for (Dimension dimension : parsedMetric.getDimensions()) {
            dimensionValues.add(dimension.getValue());
        }
        Assert.assertEquals(Arrays.asList("mnist", "Model"), dimensionValues);
        Assert.assertEquals(null, parsedMetric.getType());
        Assert.assertEquals("test-host", parsedMetric.getHostName());
        Assert.assertEquals("1699061430", parsedMetric.getTimestamp());
        Assert.assertEquals("6d1726a7-172c-4010-b671-01d71bacc451", parsedMetric.getRequestId());
    }

    @Test
    public void testParseBackendMetricLogWithType() {
        String backendMetricLog =
                "PredictionTime.Milliseconds:71.95|#ModelName:mnist,Level:Model|#type:GAUGE|#hostname:test-host,1699061430,6d1726a7-172c-4010-b671-01d71bacc451";
        Metric parsedMetric = Metric.parse(backendMetricLog);

        Assert.assertEquals("PredictionTime", parsedMetric.getMetricName());
        Assert.assertEquals("Milliseconds", parsedMetric.getUnit());
        Assert.assertEquals("71.95", parsedMetric.getValue());
        List<String> dimensionNames = new ArrayList<String>();
        for (Dimension dimension : parsedMetric.getDimensions()) {
            dimensionNames.add(dimension.getName());
        }
        Assert.assertEquals(Arrays.asList("ModelName", "Level"), dimensionNames);
        List<String> dimensionValues = new ArrayList<String>();
        for (Dimension dimension : parsedMetric.getDimensions()) {
            dimensionValues.add(dimension.getValue());
        }
        Assert.assertEquals(Arrays.asList("mnist", "Model"), dimensionValues);
        Assert.assertEquals("GAUGE", parsedMetric.getType());
        Assert.assertEquals("test-host", parsedMetric.getHostName());
        Assert.assertEquals("1699061430", parsedMetric.getTimestamp());
        Assert.assertEquals("6d1726a7-172c-4010-b671-01d71bacc451", parsedMetric.getRequestId());
    }
}
