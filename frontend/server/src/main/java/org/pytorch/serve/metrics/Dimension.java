package org.pytorch.serve.metrics;

import com.google.gson.annotations.SerializedName;
import org.pytorch.serve.servingsdk.metrics.BaseDimension;

public class Dimension implements BaseDimension {

    @SerializedName("Name")
    private String name;

    @SerializedName("Value")
    private String value;

    public Dimension() {}

    public Dimension(String name, String value) {
        this.name = name;
        this.value = value;
    }

    @Override
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @Override
    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
