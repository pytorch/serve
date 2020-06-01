

package org.pytorch.serve.servingsdk;

/**
 * Runtime exception for custom model server endpoint plugins
 */
public class ModelServerEndpointException extends RuntimeException {
    public ModelServerEndpointException(String err) {super(err);}
    public ModelServerEndpointException(String err, Throwable t) {super(err, t);}
}
