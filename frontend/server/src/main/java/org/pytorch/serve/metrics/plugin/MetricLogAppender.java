package org.pytorch.serve.metrics.plugin;

import org.apache.log4j.AppenderSkeleton;
import org.apache.log4j.spi.LoggingEvent;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MetricLogAppender extends AppenderSkeleton {

    private MetricEventPublisherImpl publisher = MetricEventPublisherImpl.getInstance();
    private Logger logger = LoggerFactory.getLogger(MetricLogAppender.class);

    @Override
    protected void append(LoggingEvent le) {
        try {
            publisher.broadcast(le);
        } catch(Exception e){
            logger.error("Ignoring the error occurred while handling the log event", e);
        }
    }

    @Override
    public void close() {
    }

    @Override
    public boolean requiresLayout() {
        return false;
    }

}