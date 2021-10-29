package org.pytorch.serve.servingsdk.metrics;

class MetricPluginException extends RuntimeException {
    public MetricPluginException(String err) {super(err);}
    public MetricPluginException(String err, Throwable t) {super(err, t);}
}