package org.pytorch.serve.metrics.configuration;

import java.util.List;

public class MetricSpecification {
    private String name;
    private String unit;
    private List<String> dimensions;

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return this.name;
    }

    public void setUnit(String unit) {
        this.unit = unit;
    }

    public String getUnit() {
        return this.unit;
    }

    public void setDimensions(List<String> dimensions) {
        this.dimensions = dimensions;
    }

    public List<String> getDimensions() {
        return this.dimensions;
    }

    @Override
    public String toString() {
        return "name: " + this.name + ", unit: " + this.unit + ", dimensions: " + this.dimensions;
    }

    public void validate() {
        if (this.name == null || this.name.isEmpty()) {
            throw new RuntimeException("Metric name cannot be empty. " + this);
        }

        if (this.unit == null || this.unit.isEmpty()) {
            throw new RuntimeException("Metric unit cannot be empty. " + this);
        }
    }
}
