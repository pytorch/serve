package org.pytorch.serve.util.messages;

import java.nio.charset.StandardCharsets;

public class InputParameter {

    private String name;
    private byte[] value;
    private CharSequence contentType;

    public InputParameter() {}

    public InputParameter(String name, String value) {
        this.name = name;
        this.value = value.getBytes(StandardCharsets.UTF_8);
    }

    public InputParameter(String name, byte[] data) {
        this(name, data, null);
    }

    public InputParameter(String name, byte[] data, CharSequence contentType) {
        this.name = name;
        this.contentType = contentType;
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
}
