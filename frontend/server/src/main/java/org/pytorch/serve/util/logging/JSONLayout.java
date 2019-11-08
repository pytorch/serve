package org.pytorch.serve.util.logging;

import org.apache.log4j.PatternLayout;
import org.apache.log4j.spi.LoggingEvent;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.util.JsonUtils;

public class JSONLayout extends PatternLayout {

    @Override
    public String format(LoggingEvent event) {
        Object eventMessage = event.getMessage();
        if (eventMessage == null) {
            return null;
        }
        if (eventMessage instanceof Metric) {
            Metric metric = (Metric) event.getMessage();
            return JsonUtils.GSON_PRETTY.toJson(metric) + '\n';
        }
        return eventMessage.toString() + '\n';
    }
}
