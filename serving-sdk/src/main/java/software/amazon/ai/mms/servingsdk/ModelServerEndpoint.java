/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package software.amazon.ai.mms.servingsdk;

import software.amazon.ai.mms.servingsdk.http.Request;
import software.amazon.ai.mms.servingsdk.http.Response;

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
