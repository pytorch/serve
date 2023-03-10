package org.pytorch.serve.metrics;

import java.util.Arrays;
import java.util.ArrayList;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import io.prometheus.client.CollectorRegistry;
import org.testng.annotations.Test;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.AfterSuite;
import org.testng.Assert;
import org.pytorch.serve.metrics.format.prometheous.PrometheusCounter;
import org.pytorch.serve.metrics.format.prometheous.PrometheusGauge;

public class MetricTest {
    private static final String testMetricName = "TestMetric";
    private static final String testMetricUnit = "ms";
    private static final ArrayList<String> testMetricDimensionNames = new ArrayList<String>(Arrays.asList("ModelName", "Level"));
    private static final ArrayList<String> testMetricDimensionValues = new ArrayList<String>(Arrays.asList("TestModel", "Model"));
    private static final String testHostname = "TestHost";
    private static final String testRequestId = "fa8639a8-d3fa-4a25-a80f-24463863fe0f";
    private static final String testTimestamp = "1678152573";
    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();
    private final PrintStream originalOut = System.out;

    @BeforeSuite
    public void setupOutStream() {
        System.setOut(new PrintStream(outContent));
    }

    @BeforeMethod
    public void clearOutStream() {
        System.out.flush();
        outContent.reset();
    }

    @BeforeMethod
    public void clearPrometheusRegistry() {
        CollectorRegistry.defaultRegistry.clear();
    }

    @AfterSuite
    public void restoreOutStream() {
        System.setOut(originalOut);
    }

    @Test
    public void testBackendLogMetric() {
        IMetric testMetric = MetricBuilder.build(
                MetricBuilder.MetricMode.LOG,
                MetricBuilder.MetricContext.BACKEND,
                MetricBuilder.MetricType.COUNTER,
                testMetricName,
                testMetricUnit,
                testMetricDimensionNames
        );
        Assert.assertEquals(testMetric.getClass(), LogMetric.class);

        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        String expectedMetricString = "MODEL_METRICS - TestMetric.ms:1.0|#ModelName:TestModel,Level:Model";
        System.out.flush();
        Assert.assertTrue(outContent.toString().contains(expectedMetricString));

        this.clearOutStream();

        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, testRequestId, testTimestamp, 1.0);
        System.out.flush();
        expectedMetricString = "MODEL_METRICS - TestMetric.ms:1.0|#ModelName:TestModel,Level:Model|#hostname:TestHost,requestID:fa8639a8-d3fa-4a25-a80f-24463863fe0f,timestamp:1678152573";
        Assert.assertTrue(outContent.toString().contains(expectedMetricString));
    }

    @Test
    public void testFrontendLogMetric() {
        IMetric testMetric = MetricBuilder.build(
                MetricBuilder.MetricMode.LOG,
                MetricBuilder.MetricContext.FRONTEND,
                MetricBuilder.MetricType.GAUGE,
                testMetricName,
                testMetricUnit,
                testMetricDimensionNames
        );
        Assert.assertEquals(testMetric.getClass(), LogMetric.class);

        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        String expectedMetricString = "TS_METRICS - TestMetric.ms:1.0|#ModelName:TestModel,Level:Model";
        System.out.flush();
        Assert.assertTrue(outContent.toString().contains(expectedMetricString));

        this.clearOutStream();

        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, testRequestId, testTimestamp, 1.0);
        System.out.flush();
        expectedMetricString = "TS_METRICS - TestMetric.ms:1.0|#ModelName:TestModel,Level:Model|#hostname:TestHost,requestID:fa8639a8-d3fa-4a25-a80f-24463863fe0f,timestamp:1678152573";
        Assert.assertTrue(outContent.toString().contains(expectedMetricString));
    }

    @Test
    public void testBackendPrometheusCounter() {
        IMetric testMetric = MetricBuilder.build(
                MetricBuilder.MetricMode.PROMETHEUS,
                MetricBuilder.MetricContext.BACKEND,
                MetricBuilder.MetricType.COUNTER,
                testMetricName,
                testMetricUnit,
                testMetricDimensionNames
        );
        Assert.assertEquals(testMetric.getClass(), PrometheusCounter.class);

        String[] dimensionNames = {testMetricDimensionNames.get(0), testMetricDimensionNames.get(1), "Hostname"};
        String[] dimensionValues = {testMetricDimensionValues.get(0), testMetricDimensionValues.get(1), testHostname};
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 1.0);
        Double metricValue = CollectorRegistry.defaultRegistry.getSampleValue(testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, new Double(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 2.0);
        metricValue = CollectorRegistry.defaultRegistry.getSampleValue(testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, new Double(3.0));
    }

    @Test
    public void testFrontendPrometheusCounter() {
        IMetric testMetric = MetricBuilder.build(
                MetricBuilder.MetricMode.PROMETHEUS,
                MetricBuilder.MetricContext.FRONTEND,
                MetricBuilder.MetricType.COUNTER,
                testMetricName,
                testMetricUnit,
                testMetricDimensionNames
        );
        Assert.assertEquals(testMetric.getClass(), PrometheusCounter.class);

        String[] dimensionNames = {testMetricDimensionNames.get(0), testMetricDimensionNames.get(1)};
        String[] dimensionValues = {testMetricDimensionValues.get(0), testMetricDimensionValues.get(1)};
        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        Double metricValue = CollectorRegistry.defaultRegistry.getSampleValue(testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, new Double(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, 2.0);
        metricValue = CollectorRegistry.defaultRegistry.getSampleValue(testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, new Double(3.0));
    }

    @Test
    public void testBackendPrometheusGauge() {
        IMetric testMetric = MetricBuilder.build(
                MetricBuilder.MetricMode.PROMETHEUS,
                MetricBuilder.MetricContext.BACKEND,
                MetricBuilder.MetricType.GAUGE,
                testMetricName,
                testMetricUnit,
                testMetricDimensionNames
        );
        Assert.assertEquals(testMetric.getClass(), PrometheusGauge.class);

        String[] dimensionNames = {testMetricDimensionNames.get(0), testMetricDimensionNames.get(1), "Hostname"};
        String[] dimensionValues = {testMetricDimensionValues.get(0), testMetricDimensionValues.get(1), testHostname};
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 1.0);
        Double metricValue = CollectorRegistry.defaultRegistry.getSampleValue(testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, new Double(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, testHostname, null, null, 2.0);
        metricValue = CollectorRegistry.defaultRegistry.getSampleValue(testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, new Double(2.0));
    }

    @Test
    public void testFrontendPrometheusGauge() {
        IMetric testMetric = MetricBuilder.build(
                MetricBuilder.MetricMode.PROMETHEUS,
                MetricBuilder.MetricContext.FRONTEND,
                MetricBuilder.MetricType.GAUGE,
                testMetricName,
                testMetricUnit,
                testMetricDimensionNames
        );
        Assert.assertEquals(testMetric.getClass(), PrometheusGauge.class);

        String[] dimensionNames = {testMetricDimensionNames.get(0), testMetricDimensionNames.get(1)};
        String[] dimensionValues = {testMetricDimensionValues.get(0), testMetricDimensionValues.get(1)};
        testMetric.addOrUpdate(testMetricDimensionValues, 1.0);
        Double metricValue = CollectorRegistry.defaultRegistry.getSampleValue(testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, new Double(1.0));
        testMetric.addOrUpdate(testMetricDimensionValues, 2.0);
        metricValue = CollectorRegistry.defaultRegistry.getSampleValue(testMetricName, dimensionNames, dimensionValues);
        Assert.assertEquals(metricValue, new Double(2.0));
    }
}
