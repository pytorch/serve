package org.pytorch.serve.openapi;

public class QueryParameter extends Parameter {

    public QueryParameter() {
        this(null, "string", null, false, null);
    }

    public QueryParameter(String name, String description) {
        this(name, "string", null, false, description);
    }

    public QueryParameter(String name, String type, String description) {
        this(name, type, null, false, description);
    }

    public QueryParameter(String name, String type, String defaultValue, String description) {
        this(name, type, defaultValue, false, description);
    }

    public QueryParameter(
            String name, String type, String defaultValue, boolean required, String description) {
        this.name = name;
        this.description = description;
        in = "query";
        this.required = required;
        schema = new Schema(type, null, defaultValue);
    }
}
