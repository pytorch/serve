package org.pytorch.serve.servingsdk.impl;

import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.pytorch.serve.servingsdk.http.Request;

public class ModelServerRequest implements Request {
    private FullHttpRequest req;
    private QueryStringDecoder decoder;

    public ModelServerRequest(FullHttpRequest r, QueryStringDecoder d) {
        req = r;
        decoder = d;
    }

    @Override
    public List<String> getHeaderNames() {
        return new ArrayList<>(req.headers().names());
    }

    @Override
    public String getRequestURI() {
        return req.uri();
    }

    @Override
    public Map<String, List<String>> getParameterMap() {
        return decoder.parameters();
    }

    @Override
    public List<String> getParameter(String k) {
        return decoder.parameters().get(k);
    }

    @Override
    public String getContentType() {
        return HttpUtil.getMimeType(req).toString();
    }

    @Override
    public ByteArrayInputStream getInputStream() {
        return new ByteArrayInputStream(req.content().array());
    }
}
