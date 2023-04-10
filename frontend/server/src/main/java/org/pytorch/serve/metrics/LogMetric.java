package org.pytorch.serve.metrics;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import org.pytorch.serve.util.ConfigManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogMetric extends IMetric {
    /**
     * Note: hostname, timestamp, and requestid(if available) are automatically added in log metric.
     */
    private static final Logger loggerTsMetrics =
            LoggerFactory.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);

    private static final Logger loggerModelMetrics =
            LoggerFactory.getLogger(ConfigManager.MODEL_METRICS_LOGGER);

    public LogMetric(
            MetricBuilder.MetricType type, String name, String unit, List<String> dimensionNames) {
        super(type, name, unit, dimensionNames);
    }

    @Override
    public void addOrUpdate(List<String> dimensionValues, double value) {
        // Used for logging frontend metrics
        String metricString = this.buildMetricString(dimensionValues, value);
        loggerTsMetrics.info(metricString);
    }

    @Override
    public void addOrUpdate(List<String> dimensionValues, String requestIds, double value) {
        // Used for logging backend metrics
        String metricString = this.buildMetricString(dimensionValues, requestIds, value);
        loggerModelMetrics.info(metricString);
    }

    private String buildMetricString(List<String> dimensionValues, double value) {
        StringBuilder metricStringBuilder = new StringBuilder();
        metricStringBuilder
                .append(this.name)
                .append('.')
                .append(this.unit)
                .append(':')
                .append(value)
                .append("|#");

        // Exclude the final dimension which is expected to be Hostname
        int dimensionsCount = Math.min(this.dimensionNames.size() - 1, dimensionValues.size() - 1);
        List<String> dimensions = new ArrayList<String>();
        for (int index = 0; index < dimensionsCount; index++) {
            dimensions.add(this.dimensionNames.get(index) + ":" + dimensionValues.get(index));
        }
        metricStringBuilder.append(dimensions.stream().collect(Collectors.joining(",")));

        // The final dimension is expected to be Hostname
        metricStringBuilder
                .append("|#hostname:")
                .append(dimensionValues.get(dimensionValues.size() - 1));

        metricStringBuilder
                .append(",timestamp:")
                .append(
                        String.valueOf(
                                TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis())));

        return metricStringBuilder.toString();
    }

    private String buildMetricString(
            List<String> dimensionValues, String requestIds, double value) {
        StringBuilder metricStringBuilder = new StringBuilder();
        metricStringBuilder
                .append(this.name)
                .append('.')
                .append(this.unit)
                .append(':')
                .append(value)
                .append("|#");

        // Exclude the final dimension which is expected to be Hostname
        int dimensionsCount = Math.min(this.dimensionNames.size() - 1, dimensionValues.size() - 1);
        List<String> dimensions = new ArrayList<String>();
        for (int index = 0; index < dimensionsCount; index++) {
            dimensions.add(this.dimensionNames.get(index) + ":" + dimensionValues.get(index));
        }
        metricStringBuilder.append(dimensions.stream().collect(Collectors.joining(",")));

        // The final dimension is expected to be Hostname
        metricStringBuilder
                .append("|#hostname:")
                .append(dimensionValues.get(dimensionValues.size() - 1));

        metricStringBuilder.append(",requestID:").append(requestIds);

        metricStringBuilder
                .append(",timestamp:")
                .append(
                        String.valueOf(
                                TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis())));

        return metricStringBuilder.toString();
    }
}
