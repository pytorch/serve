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

import io.netty.buffer.ByteBufOutputStream;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.OutputStream;
import software.amazon.ai.mms.servingsdk.http.Response;

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
