package org.pytorch.serve.servingsdk;

import java.util.Date;

public interface LogEvent {
    /**
     * Get the log level
     * @return The name of this model
     */
    String getLevel();

    /**
     * Get the name of this model
     * @return The name of this model
     */
    String getMessage();

    /**
     * Get the name of this model
     * @return The name of this model
     */
    Date getTimestamp();
}


