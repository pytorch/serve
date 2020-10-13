package org.pytorch.serve.metrics.plugin;
import java.util.Date;

import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.servingsdk.metrics.BaseMetric;
import org.pytorch.serve.servingsdk.metrics.MetricLogEvent;

public class MetricLogEventImpl implements MetricLogEvent  {

    private String level;
    private String message;
    private Date timestamp;
    private BaseMetric metric;

    public MetricLogEventImpl(String level, String message, Date timestamp) {
        this.level = level;
        this.message = message;
        this.timestamp = timestamp;
        this.metric = Metric.parse(message);
    }

    public String getLevel() {
        return level;
    }

    @Override
    public BaseMetric getMetric() {
        return metric;
    }

    public void setLevel(String level) {
        this.level = level;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Date getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Date timestamp) {
        this.timestamp = timestamp;
    }


}

