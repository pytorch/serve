package org.pytorch.serve.http;

public class ResourceNotFoundException extends RuntimeException {

    static final long serialVersionUID = 1L;

    /**
     * Constructs an {@code ResourceNotFoundException} with {@code null} as its error detail
     * message.
     */
    public ResourceNotFoundException() {
        super("Requested resource is not found, please refer to API document.");
    }
}
