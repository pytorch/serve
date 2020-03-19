package org.pytorch.serve.util.messages;

import java.util.Map;

public class Predictions {

    private String requestId;
    private int statusCode;
    private String reasonPhrase;

    private String contentType;
    private Map<String, String> headers;
    private byte[] resp;

    public Map<String, String> getHeaders() {
        return headers;
    }

    public void setHeaders(Map<String, String> headers) {
        this.headers = headers;
    }

    public Predictions() {}

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public byte[] getResp() {
        return resp;
    }

    public void setResp(byte[] resp) {
        this.resp = resp.clone();
    }

    public String getContentType() {
        return contentType;
    }

    public void setStatusCode(int statusCode) {
        this.statusCode = statusCode;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public int getStatusCode() {
        return statusCode;
    }

    public String getReasonPhrase() {
        return reasonPhrase;
    }

    public void setReasonPhrase(String reasonPhrase) {
        this.reasonPhrase = reasonPhrase;
    }
}
