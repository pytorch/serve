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

package com.amazonaws.ml.mms.servingsdk.impl;

import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import software.amazon.ai.mms.servingsdk.http.Request;

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
