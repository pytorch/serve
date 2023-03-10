package org.pytorch.serve.metrics;

import java.util.ArrayList;
import org.pytorch.serve.util.ConfigManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LogMetric extends IMetric {
    /**
     * Note:
     * hostname, timestamp, and requestid(if available) are automatically added in log metric.
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
        this.addOrUpdate(dimensionValues, null, null, null, value);
    }

    @Override
    public void addOrUpdate(
            ArrayList<String> dimensionValues, String hostname, String timestamp, String requestIds, double value) {
        String metricString = this.buildMetricString(dimensionValues, hostname, timestamp, requestIds, value);
        this.emitMetricLog(metricString);
    }

    private String buildMetricString(
            ArrayList<String> dimensionValues, String hostname, String requestIds, String timestamp, double value) {
        StringBuilder sb = new StringBuilder(128);
        sb.append(this.name).append('.').append(this.unit).append(':').append(value).append("|#");

        boolean first = true;
        for (int index=0; index<Math.min(this.dimensionNames.size(), dimensionValues.size()); index++) {
            if (first) {
                first = false;
            } else {
                sb.append(',');
            }
            sb.append(this.dimensionNames.get(index)).append(':').append(dimensionValues.get(index));
        }
        if (hostname != null && !hostname.isEmpty()) {
            sb.append("|#hostname:").append(hostname);
        }
        if (requestIds != null && !requestIds.isEmpty()) {
            sb.append(",requestID:").append(requestIds);
        }
        if (timestamp != null && !timestamp.isEmpty()) {
            sb.append(",timestamp:").append(timestamp);
        }

        return sb.toString();
    }

    private void emitMetricLog(String metricString) {
        if (this.context == MetricBuilder.MetricContext.BACKEND) {
            loggerModelMetrics.info(metricString);
        } else {
            loggerTsMetrics.info(metricString);
        }
    }
}
