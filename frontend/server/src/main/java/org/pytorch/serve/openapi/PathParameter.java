package org.pytorch.serve.openapi;

public class PathParameter extends Parameter {

    public PathParameter() {
        this(null, "string", null, null);
    }

    public PathParameter(String name, String description) {
        this(name, "string", null, description);
    }

    public PathParameter(String name, String type, String defaultValue, String description) {
        this.name = name;
        this.description = description;
        in = "path";
        required = true;
        schema = new Schema(type, null, defaultValue);
    }
}
