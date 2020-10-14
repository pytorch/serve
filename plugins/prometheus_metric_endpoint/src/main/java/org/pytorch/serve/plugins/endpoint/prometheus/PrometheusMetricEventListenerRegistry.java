package org.pytorch.serve.plugins.endpoint.prometheus;

import org.pytorch.serve.servingsdk.metrics.*;
import java.util.List;

/**
 * This class extends MetricEventListenerRegistry from Torch Serve SDK to register listener with publisher.
 * At the time of initialization of Torch Serve server, the class gets loaded and register method
 * is invoked with the MetricEventPublisher object.
 */
public class PrometheusMetricEventListenerRegistry extends MetricEventListenerRegistry {
    /**
     * Registers the Metric Event Listener to the publisher
     *
     * @param publisher MetricEventPublisher object
     */
    public void register(MetricEventPublisher publisher) {
        PrometheusMetricEventListener listener = new PrometheusMetricEventListener();
        publisher.addMetricEventListener(listener);
    }
}

