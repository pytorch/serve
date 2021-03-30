package org.pytorch.serve.archive.utils;

public class InvalidArchiveURLException extends Exception {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs an {@code InvalidArchiveURLException} with the specified detail message.
     *
     * @param message The detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method)
     */
    public InvalidArchiveURLException(String message) {
        super(message);
    }

    /**
     * Constructs an {@code InvalidArchiveURLException} with the specified detail message and cause.
     *
     * <p>Note that the detail message associated with {@code cause} is <i>not</i> automatically
     * incorporated into this exception's detail message.
     *
     * @param message The detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method)
     * @param cause The cause (which is saved for later retrieval by the {@link #getCause()}
     *     method). (A null value is permitted, and indicates that the cause is nonexistent or
     *     unknown.)
     */
    public InvalidArchiveURLException(String message, Throwable cause) {
        super(message, cause);
    }
}
