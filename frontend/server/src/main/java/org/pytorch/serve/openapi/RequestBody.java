package org.pytorch.serve.openapi;

import java.util.LinkedHashMap;
import java.util.Map;

public class RequestBody {

    private String description;
    private Map<String, MediaType> content;
    private boolean required;

    public RequestBody() {}

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Map<String, MediaType> getContent() {
        return content;
    }

    public void setContent(Map<String, MediaType> content) {
        this.content = content;
    }

    public void addContent(MediaType mediaType) {
        if (content == null) {
            content = new LinkedHashMap<>();
        }
        content.put(mediaType.getContentType(), mediaType);
    }

    public boolean isRequired() {
        return required;
    }

    public void setRequired(boolean required) {
        this.required = required;
    }
}
