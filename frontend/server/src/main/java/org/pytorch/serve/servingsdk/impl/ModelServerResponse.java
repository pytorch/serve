package org.pytorch.serve.servingsdk.impl;

import io.netty.buffer.ByteBufOutputStream;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.OutputStream;
import org.pytorch.serve.servingsdk.http.Response;

public class ModelServerResponse implements Response {

    private FullHttpResponse response;

    public ModelServerResponse(FullHttpResponse rsp) {
        response = rsp;
    }

    @Override
    public void setStatus(int i) {
        response.setStatus(HttpResponseStatus.valueOf(i));
    }

    @Override
    public void setStatus(int i, String s) {
        response.setStatus(HttpResponseStatus.valueOf(i, s));
    }

    @Override
    public void setHeader(String k, String v) {
        response.headers().set(k, v);
    }

    @Override
    public void addHeader(String k, String v) {
        response.headers().add(k, v);
    }

    @Override
    public void setContentType(String contentType) {
        response.headers().set(HttpHeaderNames.CONTENT_TYPE, contentType);
    }

    @Override
    public OutputStream getOutputStream() {
        return new ByteBufOutputStream(response.content());
    }
}
