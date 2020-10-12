package org.pytorch.serve.metrics.plugin;

import org.apache.log4j.AppenderSkeleton;
import org.apache.log4j.spi.LoggingEvent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class InternalAppender extends AppenderSkeleton {

    private SingletonAppenderImpl appender = SingletonAppenderImpl.getInstance();
    private Logger logger = LoggerFactory.getLogger(InternalAppender.class);

    @Override
    protected void append(LoggingEvent le) {
        try {
            appender.append(le);
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