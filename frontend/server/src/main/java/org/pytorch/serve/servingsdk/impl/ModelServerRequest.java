package org.pytorch.serve.servingsdk.impl;

import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpConstants;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.util.AsciiString;
import io.netty.util.internal.ObjectUtil;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.util.NettyUtils;

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

    public String[] getContentEncoding() {
        CharSequence contentEncodingValue = req.headers().get(HttpHeaderNames.CONTENT_ENCODING);
        if (contentEncodingValue != null) {
            return splitContentEncodings(contentEncodingValue);
        } else {
            return null;
        }
    }

    @Override
    public ByteArrayInputStream getInputStream() {
        return new ByteArrayInputStream(NettyUtils.getBytes(req.content()));
    }

    private String[] splitContentEncodings(CharSequence contentEncodingValue) {
        ObjectUtil.checkNotNull(contentEncodingValue, "contentEncodingValue");

        AsciiString[] encodings =
                new AsciiString(contentEncodingValue).split((char) HttpConstants.COMMA);

        String[] result = new String[encodings.length];
        for (int i = 0; i != encodings.length; i++) {
            result[i] = encodings[i].trim().toString();
        }
        return result;
    }
}
