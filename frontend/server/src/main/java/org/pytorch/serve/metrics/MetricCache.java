package org.pytorch.serve.metrics;

import java.util.Map;
import java.util.concurrent.ConcurrentMap;

public class MetricCache {
    Map<String, IMetric> metricsFrontend;
    ConcurrentMap<String, IMetric> metricsBackend;

    public void addMetricFrontend(String metricName, IMetric metric);

    public IMetric getMetricFrontend(String metricName);

    public void addMetricBackend(String metricName, IMetric metric);

    public IMetric getMetricBackend(String metricName);
}
