package org.pytorch.serve.test.plugins.metrics;

import org.pytorch.serve.servingsdk.metrics.MetricEventPublisher;
import org.pytorch.serve.servingsdk.metrics.MetricEventListenerRegistry;


public class TestMetricEventListenerRegistry extends MetricEventListenerRegistry {
    public void register(MetricEventPublisher publisher) {
        TestMetricEventListener listener = new TestMetricEventListener();
        publisher.addMetricEventListener(listener);
    }
}

