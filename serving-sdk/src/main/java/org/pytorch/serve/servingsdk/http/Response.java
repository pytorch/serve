

package org.pytorch.serve.servingsdk.http;

import java.io.IOException;
import java.io.OutputStream;

/**
 * Interface defining the response object sent to the custom defined endpoints
 */
public interface Response {
    /**
     * Set HTTP response status
     * @param sc - status code
     */
    void setStatus(int sc);

    /**
     * Set HTTP response status code and status phrase
     * @param sc - Integer value representing the status code of this response
     * @param phrase - String phrase representing the status phrase of this response
     */
    void setStatus(int sc, String phrase);

    /**
     * Set HTTP headers
     * @param k - Header name
     * @param v - Header value
     */
    void setHeader(String k, String v);

    /**
     * Add HTTP headers for an existing header name
     * @param k - Header name
     * @param v - Header value
     */
    void addHeader(String k, String v);

    /**
     * Set content type header in the response object
     * @param ct - Content-Type
     */
    void setContentType(String ct);

    /**
     * Get the output stream object for response
     * @return response body content as OutputStream
     * @throws IOException if I/O error occurs
     */
    OutputStream getOutputStream() throws IOException;
}
