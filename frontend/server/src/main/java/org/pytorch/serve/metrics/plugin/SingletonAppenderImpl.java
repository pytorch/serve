package org.pytorch.serve.metrics.plugin;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import org.apache.log4j.spi.LoggingEvent;

import org.pytorch.serve.servingsdk.CachingSingletonAppender;
import org.pytorch.serve.servingsdk.LogEventListener;



public class SingletonAppenderImpl implements  CachingSingletonAppender {

    private static final SingletonAppenderImpl theInstance = new SingletonAppenderImpl();
    private List<LogEventListener> listeners;

    private SingletonAppenderImpl() {
        listeners = new ArrayList<>();
    }

    public static SingletonAppenderImpl getInstance() {
        return theInstance;
    }

    public void append(LoggingEvent le) {
        LogEventImpl event = new LogEventImpl(le.getLevel().toString(), le.getMessage().toString(), new Date(le.getTimeStamp()));
        if (!listeners.isEmpty()) {
            for (LogEventListener listener : listeners) {
                listener.handle(event);
            }
        }
    }

    @Override
    public void addLoggingEventListener(LogEventListener listener) {
        listeners.add(listener);
    }

    @Override
    public void removeLoggingEventListener(LogEventListener listener) {
        listeners.remove(listener);
    }

}