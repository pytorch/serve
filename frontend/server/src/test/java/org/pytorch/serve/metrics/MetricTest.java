package org.pytorch.serve.metrics;

import io.prometheus.client.CollectorRegistry;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
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
            new ArrayList<String>(Arrays.asList("ModelName", "Level"));
    private final ArrayList<String> testMetricDimensionValues =
            new ArrayList<String>(Arrays.asList("TestModel", "Model"));
    private final String testHostname = "TestHost";
    private final String testRequestId = "fa8639a8-d3fa-4a25-a80f-24463863fe0f";
    private final String testTimestamp = "1678152573";
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
                        MetricBuilder.MetricContext.BACKEND,
                        MetricBuilder.MetricType.COUNTER,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), LogMetric.class);

        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        String expectedMetricString = "TestMetric.ms:1.0|#ModelName:TestModel,Level:Model";
        Assert.assertTrue(modelMetricsContent.toString().contains(expectedMetricString));

        this.flushLogWriterStreams();

        testMetric.addOrUpdate(
                testMetricDimensionValues, testHostname, testRequestId, testTimestamp, 1.0);
        expectedMetricString =
                "TestMetric.ms:1.0|#ModelName:TestModel,Level:Model|#hostname:TestHost,"
                        + "requestID:fa8639a8-d3fa-4a25-a80f-24463863fe0f,timestamp:1678152573";
        Assert.assertTrue(modelMetricsContent.toString().contains(expectedMetricString));
    }

    @Test
    public void testFrontendLogMetric() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.LOG,
                        MetricBuilder.MetricContext.FRONTEND,
                        MetricBuilder.MetricType.GAUGE,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), LogMetric.class);

        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        String expectedMetricString = "TestMetric.ms:1.0|#ModelName:TestModel,Level:Model";
        Assert.assertTrue(tsMetricsContent.toString().contains(expectedMetricString));

        this.flushLogWriterStreams();

        testMetric.addOrUpdate(
                testMetricDimensionValues, testHostname, testRequestId, testTimestamp, 1.0);
        expectedMetricString =
                "TestMetric.ms:1.0|#ModelName:TestModel,Level:Model|#hostname:TestHost,"
                        + "requestID:fa8639a8-d3fa-4a25-a80f-24463863fe0f,timestamp:1678152573";
        Assert.assertTrue(tsMetricsContent.toString().contains(expectedMetricString));
    }

    @Test
    public void testBackendPrometheusCounter() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricContext.BACKEND,
                        MetricBuilder.MetricType.COUNTER,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusCounter.class);

        String[] dimensionNames = {
            testMetricDimensionNames.get(0), testMetricDimensionNames.get(1), "Hostname"
        };
        String[] dimensionValues = {
            testMetricDimensionValues.get(0), testMetricDimensionValues.get(1), testHostname
        };
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(3.0));
    }

    @Test
    public void testFrontendPrometheusCounter() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricContext.FRONTEND,
                        MetricBuilder.MetricType.COUNTER,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusCounter.class);

        String[] dimensionNames = {
            testMetricDimensionNames.get(0), testMetricDimensionNames.get(1)
        };
        String[] dimensionValues = {
            testMetricDimensionValues.get(0), testMetricDimensionValues.get(1)
        };
        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(3.0));
    }

    @Test
    public void testBackendPrometheusGauge() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricContext.BACKEND,
                        MetricBuilder.MetricType.GAUGE,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusGauge.class);

        String[] dimensionNames = {
            testMetricDimensionNames.get(0), testMetricDimensionNames.get(1), "Hostname"
        };
        String[] dimensionValues = {
            testMetricDimensionValues.get(0), testMetricDimensionValues.get(1), testHostname
        };
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(2.0));
    }

    @Test
    public void testFrontendPrometheusGauge() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricContext.FRONTEND,
                        MetricBuilder.MetricType.GAUGE,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusGauge.class);

        String[] dimensionNames = {
            testMetricDimensionNames.get(0), testMetricDimensionNames.get(1)
        };
        String[] dimensionValues = {
            testMetricDimensionValues.get(0), testMetricDimensionValues.get(1)
        };
        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(2.0));
    }

    @Test
    public void testBackendPrometheusHistogram() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricContext.BACKEND,
                        MetricBuilder.MetricType.HISTOGRAM,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusHistogram.class);

        String[] dimensionNames = {
            testMetricDimensionNames.get(0), testMetricDimensionNames.get(1), "Hostname"
        };
        String[] dimensionValues = {
            testMetricDimensionValues.get(0), testMetricDimensionValues.get(1), testHostname
        };
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName + "_sum", dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName + "_sum", dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(3.0));
    }

    @Test
    public void testFrontendPrometheusHistogram() {
        IMetric testMetric =
                MetricBuilder.build(
                        MetricBuilder.MetricMode.PROMETHEUS,
                        MetricBuilder.MetricContext.FRONTEND,
                        MetricBuilder.MetricType.HISTOGRAM,
                        testMetricName,
                        testMetricUnit,
                        testMetricDimensionNames);
        Assert.assertEquals(testMetric.getClass(), PrometheusHistogram.class);

        String[] dimensionNames = {
            testMetricDimensionNames.get(0), testMetricDimensionNames.get(1)
        };
        String[] dimensionValues = {
            testMetricDimensionValues.get(0), testMetricDimensionValues.get(1)
        };
        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        Double metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName + "_sum", dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, 2.0);
        metricValue =
                CollectorRegistry.defaultRegistry.getSampleValue(
                        testMetricName + "_sum", dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, Double.valueOf(3.0));
    }
}
