/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package com.amazonaws.ml.mms.util.codec;

import com.amazonaws.ml.mms.util.messages.BaseModelRequest;
import com.amazonaws.ml.mms.util.messages.InputParameter;
import com.amazonaws.ml.mms.util.messages.ModelInferenceRequest;
import com.amazonaws.ml.mms.util.messages.ModelLoadModelRequest;
import com.amazonaws.ml.mms.util.messages.RequestInput;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToByteEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Map;

@ChannelHandler.Sharable
public class ModelRequestEncoder extends MessageToByteEncoder<BaseModelRequest> {

    @Override
    protected void encode(ChannelHandlerContext ctx, BaseModelRequest msg, ByteBuf out) {
        if (msg instanceof ModelLoadModelRequest) {
            out.writeByte('L');

            ModelLoadModelRequest request = (ModelLoadModelRequest) msg;
            byte[] buf = msg.getModelName().getBytes(StandardCharsets.UTF_8);
            out.writeInt(buf.length);
            out.writeBytes(buf);

            buf = request.getModelPath().getBytes(StandardCharsets.UTF_8);
            out.writeInt(buf.length);
            out.writeBytes(buf);

            int batchSize = request.getBatchSize();
            if (batchSize <= 0) {
                batchSize = 1;
            }
            out.writeInt(batchSize);

            buf = request.getHandler().getBytes(StandardCharsets.UTF_8);
            out.writeInt(buf.length);
            out.writeBytes(buf);

            out.writeInt(request.getGpuId());
        } else if (msg instanceof ModelInferenceRequest) {
            out.writeByte('I');
            ModelInferenceRequest request = (ModelInferenceRequest) msg;
            for (RequestInput input : request.getRequestBatch()) {
                encodeRequest(input, out);
            }
            out.writeInt(-1); // End of List
        }
    }

    private void encodeRequest(RequestInput req, ByteBuf out) {
        byte[] buf = req.getRequestId().getBytes(StandardCharsets.UTF_8);
        out.writeInt(buf.length);
        out.writeBytes(buf);

        for (Map.Entry<String, String> entry : req.getHeaders().entrySet()) {
            encodeField(entry.getKey(), out);
            encodeField(entry.getValue(), out);
        }
        out.writeInt(-1); // End of List

        for (InputParameter input : req.getParameters()) {
            encodeParameter(input, out);
        }
        out.writeInt(-1); // End of List
    }

    private void encodeParameter(InputParameter parameter, ByteBuf out) {
        byte[] modelInputName = parameter.getName().getBytes(StandardCharsets.UTF_8);
        out.writeInt(modelInputName.length);
        out.writeBytes(modelInputName);

        encodeField(parameter.getContentType(), out);

        byte[] buf = parameter.getValue();
        out.writeInt(buf.length);
        out.writeBytes(buf);
    }

    private static void encodeField(CharSequence field, ByteBuf out) {
        if (field == null) {
            out.writeInt(0);
            return;
        }
        byte[] buf = field.toString().getBytes(StandardCharsets.UTF_8);
        out.writeInt(buf.length);
        out.writeBytes(buf);
    }
}
