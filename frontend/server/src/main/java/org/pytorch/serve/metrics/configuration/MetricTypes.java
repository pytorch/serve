package org.pytorch.serve.metrics.configuration;

import java.util.List;

public class MetricTypes {
    private List<MetricSpecification> counter;
    private List<MetricSpecification> gauge;
    private List<MetricSpecification> histogram;

    public void setCounter(List<MetricSpecification> counter) {
        this.counter = counter;
    }

    public List<MetricSpecification> getCounter() {
        return this.counter;
    }

    public void setGauge(List<MetricSpecification> gauge) {
        this.gauge = gauge;
    }

    public List<MetricSpecification> getGauge() {
        return this.gauge;
    }

    public void setHistogram(List<MetricSpecification> histogram) {
        this.histogram = histogram;
    }

    public List<MetricSpecification> getHistogram() {
        return this.histogram;
    }

    public void validate() {
        if (this.counter != null) {
            for (MetricSpecification spec : this.counter) {
                spec.validate();
            }
        }

        if (this.gauge != null) {
            for (MetricSpecification spec : this.gauge) {
                spec.validate();
            }
        }

        if (this.histogram != null) {
            for (MetricSpecification spec : this.histogram) {
                spec.validate();
            }
        }
    }
}
