package org.pytorch.serve.http;

import com.google.gson.annotations.Expose;

public class StatusResponse {

    private int httpResponseCode;
    @Expose private String status;
    private Throwable e;

    public StatusResponse() {}

    public StatusResponse(String status, int httpResponseCode) {
        this.status = status;
        this.httpResponseCode = httpResponseCode;
    }

    public int getHttpResponseCode() {
        return httpResponseCode;
    }

    public void setHttpResponseCode(int httpResponseCode) {
        this.httpResponseCode = httpResponseCode;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public Throwable getE() {
        return e;
    }

    public void setE(Throwable e) {
        this.e = e;
    }
}
