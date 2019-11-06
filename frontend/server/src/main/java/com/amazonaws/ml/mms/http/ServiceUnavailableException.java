package com.amazonaws.ml.mms.http;

public class ServiceUnavailableException extends RuntimeException {

    static final long serialVersionUID = 1L;

    /**
     * Constructs an {@code ServiceUnavailableException} with the specified detail message.
     *
     * @param message The detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method)
     */
    public ServiceUnavailableException(String message) {
        super(message);
    }
}
