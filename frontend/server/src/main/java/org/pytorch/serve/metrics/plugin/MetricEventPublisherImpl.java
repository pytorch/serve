package org.pytorch.serve.metrics.plugin;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import org.apache.log4j.spi.LoggingEvent;
import org.pytorch.serve.servingsdk.metrics.MetricEventListener;
import org.pytorch.serve.servingsdk.metrics.MetricEventPublisher;

/** A class publishing Metric events to listeners */
public final class MetricEventPublisherImpl implements MetricEventPublisher {

    private static final MetricEventPublisherImpl INSTANCE = new MetricEventPublisherImpl();
    private List<MetricEventListener> listeners;

    private MetricEventPublisherImpl() {
        listeners = new ArrayList<>();
    }

    public static MetricEventPublisherImpl getInstance() {
        return INSTANCE;
    }

    public void broadcast(LoggingEvent le) {
        MetricLogEventImpl event =
                new MetricLogEventImpl(
                        le.getLevel().toString(),
                        le.getMessage().toString(),
                        new Date(le.getTimeStamp()));
        if (!listeners.isEmpty() && event.getMetric() != null) {
            for (MetricEventListener listener : listeners) {
                listener.handle(event);
            }
        }
    }

    @Override
    public void addMetricEventListener(MetricEventListener metricEventListener) {
        listeners.add(metricEventListener);
    }

    @Override
    public void removeMetricEventListener(MetricEventListener metricEventListener) {
        listeners.remove(metricEventListener);
    }
}
