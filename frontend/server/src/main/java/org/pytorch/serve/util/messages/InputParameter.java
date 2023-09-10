package org.pytorch.serve.util.messages;

import java.nio.charset.StandardCharsets;

public class InputParameter {

    private String name;
    private byte[] value;
    private CharSequence contentType;
    private String[] contentEncoding;

    public InputParameter() {}

    public InputParameter(String name, String value) {
        this.name = name;
        this.value = value.getBytes(StandardCharsets.UTF_8);
    }

    public InputParameter(String name, byte[] data) {
        this(name, data, null, null);
    }

    public InputParameter(
            String name, byte[] data, CharSequence contentType, String[] contentEncoding) {
        this.name = name;
        this.contentType = contentType;
        this.contentEncoding = contentEncoding;
        this.value = data.clone();
    }

    public String getName() {
        return name;
    }

    public byte[] getValue() {
        return value;
    }

    public CharSequence getContentType() {
        return contentType;
    }

    public String[] getContentEncoding() {
        return contentEncoding;
    }
}
