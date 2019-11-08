package org.pytorch.serve.openapi;

import java.util.LinkedHashMap;
import java.util.Map;

public class MediaType {

    private transient String contentType;
    private Schema schema;
    private Map<String, Encoding> encoding;

    public MediaType() {}

    public MediaType(String contentType, Schema schema) {
        this.contentType = contentType;
        this.schema = schema;
    }

    public String getContentType() {
        return contentType;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public Schema getSchema() {
        return schema;
    }

    public void setSchema(Schema schema) {
        this.schema = schema;
    }

    public Map<String, Encoding> getEncoding() {
        return encoding;
    }

    public void setEncoding(Map<String, Encoding> encoding) {
        this.encoding = encoding;
    }

    public void addEncoding(String contentType, Encoding encoding) {
        if (this.encoding == null) {
            this.encoding = new LinkedHashMap<>();
        }
        this.encoding.put(contentType, encoding);
    }
}
