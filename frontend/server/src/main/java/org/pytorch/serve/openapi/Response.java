package org.pytorch.serve.openapi;

import java.util.LinkedHashMap;
import java.util.Map;

public class Response {

    private transient String code;
    private String description;
    private Map<String, MediaType> content;

    public Response() {}

    public Response(String code, String description) {
        this.code = code;
        this.description = description;
    }

    public Response(String code, String description, MediaType mediaType) {
        this.code = code;
        this.description = description;
        content = new LinkedHashMap<>();
        content.put(mediaType.getContentType(), mediaType);
    }

    public String getCode() {
        return code;
    }

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
}
