

package org.pytorch.serve.servingsdk.http;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

/**
 * This defines the request object given to the custom endpoint
 */
public interface Request {
    /**
     * Get all header names in the request object
     * @return List of request header names
     */
    List<String> getHeaderNames();

    /**
     * Get the URI of the request
     * @return URI of the endpoint
     */
    String getRequestURI();

    /**
     * Get all query parameters coming in for this endpoint
     * @return a dictionary of all the parameters in the query
     */
    Map<String, List<String>> getParameterMap();

    /**
     * Get a query parameter
     * @param k - Parameter name
     * @return - value of the parameter
     */
    List<String> getParameter(String k);

    /**
     * Get the content-type of the incoming request object
     * @return content-type string in the request
     */
    String getContentType();

    /**
     * Get the body content stream of the incoming request
     * @return the request content input stream
     * @throws IOException if there is an I/O error
     */
    InputStream getInputStream() throws IOException;
}
