package org.pytorch.serve.http;

import io.netty.handler.codec.http.HttpRequest;
import java.util.UUID;

public class Session {

    private String requestId;
    private String remoteIp;
    private String method;
    private String uri;
    private String protocol;
    private int code;
    private long startTime;

    public Session(String remoteIp, HttpRequest request) {
        this.remoteIp = remoteIp;
        this.uri = request.uri();
        if (request.decoderResult().isSuccess()) {
            method = request.method().name();
            protocol = request.protocolVersion().text();
        } else {
            method = "GET";
            protocol = "HTTP/1.1";
        }
        requestId = UUID.randomUUID().toString();
        startTime = System.currentTimeMillis();
    }

    public String getRequestId() {
        return requestId;
    }

    public void setCode(int code) {
        this.code = code;
    }

    @Override
    public String toString() {
        long duration = System.currentTimeMillis() - startTime;
        return remoteIp + " \"" + method + " " + uri + ' ' + protocol + "\" " + code + ' '
                + duration;
    }
}
