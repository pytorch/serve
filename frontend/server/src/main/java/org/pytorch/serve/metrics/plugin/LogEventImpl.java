package org.pytorch.serve.metrics.plugin;
import java.util.Date;
import org.pytorch.serve.servingsdk.LogEvent;

public class LogEventImpl implements LogEvent  {

    private String level;
    private String message;
    private Date timestamp;

    public LogEventImpl(String level, String message, Date timestamp) {
        this.level = level;
        this.message = message;
        this.timestamp = timestamp;
    }

    public String getLevel() {
        return level;
    }

    public void setLevel(String level) {
        this.level = level;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Date getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Date timestamp) {
        this.timestamp = timestamp;
    }


}


