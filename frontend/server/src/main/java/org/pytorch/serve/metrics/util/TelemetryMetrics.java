package org.pytorch.serve.metrics.util;

public enum TelemetryMetrics {
    ModelServerError("ModelServerError"),
    UserScriptError("UserScriptError");
    private final String name;
    private TelemetryMetrics(String s) {
        name = s;
    }
    public boolean equalsName(String otherName) {
        // (otherName == null) check is not needed because name.equals(null) returns false
        return name.equals(otherName);
    }
    public String toString() {
        return this.name;
    }
}