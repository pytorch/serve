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

class MetricMessageImpl implements MetricMessage{
    private String modelName;
    private String modelVersion;
    private String messageType;
    private String name;
    private String value;

    public MetricMessageImpl(String modelName, String modelVersion, String messageType, String name, String value) {
        this.modelName = modelName;
        this.modelVersion = modelVersion;
        this.messageType = messageType;
        this.name = name;
        this.value = value;
    }

    @Override
    public String toString() {
        return "modelName=" + modelName +
                ", modelVersion=" + modelVersion +
                ", messageType=" + messageType +
                ", name=" + name +
                ", value=" + value  ;
    }

    public static MetricMessageImpl convert(String message){
        return new MetricMessageImpl
    }

    public String getMessageType() {
        return messageType;
    }

    public void setMessageType(String messageType) {
        this.messageType = messageType;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }


    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }
}

