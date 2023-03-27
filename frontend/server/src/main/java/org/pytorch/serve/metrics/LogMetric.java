package org.pytorch.serve.metrics;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.pytorch.serve.util.ConfigManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogMetric extends IMetric {
    /**
     * Note: hostname, timestamp, and requestid(if available) are automatically added in log metric.
     */
    private static final Logger loggerModelMetrics =
            LoggerFactory.getLogger(ConfigManager.MODEL_METRICS_LOGGER);

    private static final Logger loggerTsMetrics =
            LoggerFactory.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);

    public LogMetric(
            MetricBuilder.MetricContext context,
            MetricBuilder.MetricType type,
            String name,
            String unit,
            ArrayList<String> dimensionNames) {
        super(context, type, name, unit, dimensionNames);
    }

    @Override
    public void addOrUpdate(ArrayList<String> dimensionValues, double value) {
        // Used for logging frontend metrics
        // The final entry in dimensionValues is expected to be Hostname
        String metricString =
                this.buildMetricString(
                        dimensionValues.subList(0, dimensionValues.size() - 1),
                        dimensionValues.get(dimensionValues.size() - 1),
                        null,
                        value);
        loggerTsMetrics.info(metricString);
    }

    @Override
    public void addOrUpdate(
            ArrayList<String> dimensionValues, String hostname, String requestIds, double value) {
        // Used for logging backend metrics
        String metricString = this.buildMetricString(dimensionValues, hostname, requestIds, value);
        loggerModelMetrics.info(metricString);
    }

    private String buildMetricString(
            List<String> dimensionValues, String hostname, String requestIds, double value) {
        StringBuilder sb = new StringBuilder(128);
        sb.append(this.name).append('.').append(this.unit).append(':').append(value).append("|#");

        boolean first = true;
        for (int index = 0;
                index < Math.min(this.dimensionNames.size(), dimensionValues.size());
                index++) {
            if (first) {
                first = false;
            } else {
                sb.append(',');
            }
            sb.append(this.dimensionNames.get(index))
                    .append(':')
                    .append(dimensionValues.get(index));
        }
        if (hostname != null && !hostname.isEmpty()) {
            sb.append("|#hostname:").append(hostname);
        }
        if (requestIds != null && !requestIds.isEmpty()) {
            sb.append(",requestID:").append(requestIds);
        }
        sb.append(",timestamp:")
                .append(
                        String.valueOf(
                                TimeUnit.MILLISECONDS.toSeconds(System.currentTimeMillis())));

        return sb.toString();
    }
}
