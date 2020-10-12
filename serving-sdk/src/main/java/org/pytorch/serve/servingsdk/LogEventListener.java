package org.pytorch.serve.servingsdk;

/**
 * This provides information about the model which is currently registered with Model Server
 */

public interface LogEventListener {
    /**
     * Handle the LogEvent
     */
    void handle(LogEvent event);
}

