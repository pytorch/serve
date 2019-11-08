package org.pytorch.serve.openapi;

import com.google.gson.annotations.SerializedName;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Schema {

    private String type;
    private String format;
    private String name;
    private List<String> required;
    private Map<String, Schema> properties;
    private Schema items;
    private String description;
    private Object example;
    private Schema additionalProperties;
    private String discriminator;

    @SerializedName("enum")
    private List<String> enumeration;

    @SerializedName("default")
    private String defaultValue;

    public Schema() {}

    public Schema(String type) {
        this(type, null, null);
    }

    public Schema(String type, String description) {
        this(type, description, null);
    }

    public Schema(String type, String description, String defaultValue) {
        this.type = type;
        this.description = description;
        this.defaultValue = defaultValue;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getFormat() {
        return format;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<String> getRequired() {
        return required;
    }

    public void setRequired(List<String> required) {
        this.required = required;
    }

    public Map<String, Schema> getProperties() {
        return properties;
    }

    public void setProperties(Map<String, Schema> properties) {
        this.properties = properties;
    }

    public void addProperty(String key, Schema schema, boolean requiredProperty) {
        if (properties == null) {
            properties = new LinkedHashMap<>();
        }
        properties.put(key, schema);
        if (requiredProperty) {
            if (required == null) {
                required = new ArrayList<>();
            }
            required.add(key);
        }
    }

    public Schema getItems() {
        return items;
    }

    public void setItems(Schema items) {
        this.items = items;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Object getExample() {
        return example;
    }

    public void setExample(Object example) {
        this.example = example;
    }

    public Schema getAdditionalProperties() {
        return additionalProperties;
    }

    public void setAdditionalProperties(Schema additionalProperties) {
        this.additionalProperties = additionalProperties;
    }

    public String getDiscriminator() {
        return discriminator;
    }

    public void setDiscriminator(String discriminator) {
        this.discriminator = discriminator;
    }

    public List<String> getEnumeration() {
        return enumeration;
    }

    public void setEnumeration(List<String> enumeration) {
        this.enumeration = enumeration;
    }

    public String getDefaultValue() {
        return defaultValue;
    }

    public void setDefaultValue(String defaultValue) {
        this.defaultValue = defaultValue;
    }
}
