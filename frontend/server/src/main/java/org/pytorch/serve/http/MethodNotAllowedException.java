package org.pytorch.serve.http;

public class MethodNotAllowedException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs an {@code MethodNotAllowedException} with {@code null} as its error detail
     * message.
     */
    public MethodNotAllowedException() {
        super("Requested method is not allowed, please refer to API document.");
    }
}
