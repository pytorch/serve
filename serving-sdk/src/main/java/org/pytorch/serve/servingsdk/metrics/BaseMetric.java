package org.pytorch.serve.servingsdk.metrics;


import java.util.List;


public interface BaseMetric {

    String getHostName();

    String getRequestId();

    String getMetricName();

    String getValue();

    String getUnit();

    List<BaseDimension> getDimensions();

    String getTimestamp();
}


