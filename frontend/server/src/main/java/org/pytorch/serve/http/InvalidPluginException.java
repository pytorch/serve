package org.pytorch.serve.http;

/** InvaliPluginException is thrown when there is an error while handling a Model Server plugin */
public class InvalidPluginException extends RuntimeException {

    private static final long serialVersionUID = 1L;
    /**
     * Constructs an {@code InvalidPluginException} with {@code null} as its error detail message.
     */
    public InvalidPluginException() {
        super("Registered plugin is invalid. Please re-check the configuration and the plugins.");
    }

    /**
     * Constructs an {@code InvalidPluginException} with {@code msg} as its error detail message
     *
     * @param msg : This is the error detail message
     */
    public InvalidPluginException(String msg) {
        super(msg);
    }
}
