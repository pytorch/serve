package org.pytorch.serve.servingsdk;


public interface SingletonAppender {
    void addLoggingEventListener(LogEventListener listener);
    void removeLoggingEventListener(LogEventListener listener);
}


