package org.pytorch.serve.util.codec;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToByteEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.ModelInferenceRequest;
import org.pytorch.serve.util.messages.ModelLoadModelRequest;
import org.pytorch.serve.util.messages.RequestInput;

@ChannelHandler.Sharable
public class ModelRequestEncoder extends MessageToByteEncoder<BaseModelRequest> {
    public ModelRequestEncoder(boolean preferDirect) {
        super(preferDirect);
    }

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

            String handler = request.getHandler();
            if (handler != null) {
                buf = handler.getBytes(StandardCharsets.UTF_8);
            }

            // TODO: this might be a bug. If handler isn't specified, this
            // will repeat the model path
            out.writeInt(buf.length);
            out.writeBytes(buf);

            out.writeInt(request.getGpuId());

            String envelope = request.getEnvelope();
            if (envelope != null) {
                buf = envelope.getBytes(StandardCharsets.UTF_8);
            } else {
                buf = new byte[0];
            }

            out.writeInt(buf.length);
            out.writeBytes(buf);

            out.writeBoolean(request.isLimitMaxImagePixels());
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

        if (req.isCachedInBackend()) {
            out.writeInt(-1); // End of List
            return;
        }

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
