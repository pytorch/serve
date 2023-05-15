package org.pytorch.serve.http;

import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpRequest;
import java.util.UUID;

public class Session {

    private static final String REQUEST_ID_PREFIX = "x-request-id-prefix";

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
        HttpHeaders headers = request.headers();
        if (headers.contains(REQUEST_ID_PREFIX)) {
            // adopt header value as prefix for internal request id
            requestId = headers.getAsString(REQUEST_ID_PREFIX) + ":" + UUID.randomUUID().toString();
        } else {
            requestId = UUID.randomUUID().toString();
        }
        startTime = System.currentTimeMillis();
    }

    public Session(String remoteIp, String gRPCMethod) {
        this.remoteIp = remoteIp;
        method = "gRPC";
        protocol = "HTTP/2.0";
        this.uri = gRPCMethod;
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
