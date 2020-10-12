
package org.pytorch.serve.servingsdk;

import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.http.Response;

import java.io.IOException;

/**
 * This class defines the abstract class for ModelServerEndpoint
 */
public abstract class ModelServerEndpoint {
    /**
     * This method is called when a HTTP GET method is invoked for the defined custom model server endpoint
     * @param req - Incoming request
     * @param res - Outgoing response
     * @param ctx - ModelServer's context which defines the current model-server system information
     * @throws IOException if I/O error occurs
     */
    public void doGet(Request req, Response res, Context ctx) throws ModelServerEndpointException, IOException {
        throw new ModelServerEndpointException("No implementation found .. Default implementation invoked");
    }

    /**
     * This method is called when a HTTP PUT method is invoked for the defined custom model server endpoint
     * @param req - Incoming request
     * @param res - Outgoing response
     * @param ctx - ModelServer's context which defines the current model-server system information
     * @throws IOException if I/O error occurs
     */
    public void doPut(Request req, Response res, Context ctx) throws ModelServerEndpointException, IOException {
        throw new ModelServerEndpointException("No implementation found .. Default implementation invoked");
    }

    /**
     * This method is called when a HTTP POST method is invoked for the defined custom model server endpoint
     * @param req - Incoming request
     * @param res - Outgoing response
     * @param ctx - ModelServer's context which defines the current model-server system information
     * @throws IOException if I/O error occurs
     */
    public void doPost(Request req, Response res, Context ctx) throws ModelServerEndpointException, IOException {
        throw new ModelServerEndpointException("No implementation found .. Default implementation invoked");
    }

    /**
     * This method is called when a HTTP DELETE method is invoked for the defined custom model server endpoint
     * @param req - Incoming request
     * @param res - Outgoing response
     * @param ctx - ModelServer's context which defines the current model-server system information
     * @throws IOException if I/O error occurs
     */
    public void doDelete(Request req, Response res, Context ctx) throws ModelServerEndpointException, IOException {
        throw new ModelServerEndpointException("No implementation found .. Default implementation invoked");
    }

}
