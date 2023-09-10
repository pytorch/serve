package org.pytorch.serve.util.codec;

import io.airlift.compress.MalformedInputException;
import io.airlift.compress.zstd.ZstdDecompressor;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToByteEncoder;
import io.netty.util.AsciiString;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.ModelInferenceRequest;
import org.pytorch.serve.util.messages.ModelLoadModelRequest;
import org.pytorch.serve.util.messages.RequestInput;

@ChannelHandler.Sharable
public class ModelRequestEncoder extends MessageToByteEncoder<BaseModelRequest> {

    private static final AsciiString ZSTD = AsciiString.cached("zstd");

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
                encodeRequest(ctx, input, out);
            }
            out.writeInt(-1); // End of List
        }
    }

    private void encodeRequest(ChannelHandlerContext ctx, RequestInput req, ByteBuf out) {
        byte[] buf = req.getRequestId().getBytes(StandardCharsets.UTF_8);
        out.writeInt(buf.length);
        out.writeBytes(buf);

        if (req.isCached()) {
            out.writeInt(-1); // End of List
            out.writeInt(-1); // End of List
            return;
        }

        for (Map.Entry<String, String> entry : req.getHeaders().entrySet()) {
            encodeField(entry.getKey(), out);
            encodeField(entry.getValue(), out);
        }
        out.writeInt(-1); // End of List

        for (InputParameter input : req.getParameters()) {
            encodeParameter(ctx, input, out);
        }
        out.writeInt(-1); // End of List
        req.setCached(true);
    }

    private void encodeParameter(ChannelHandlerContext ctx, InputParameter parameter, ByteBuf out) {
        String[] contentEncodings = parameter.getContentEncoding();
        if (contentEncodings == null) {
            contentEncodings = new String[0];
        }
        byte[] parameterValue = parameter.getValue();

        if (contentEncodings.length > 1) {
            throw new RuntimeException("Currently only one content encoding is supported.");
        }

        if (contentEncodings.length == 1) {
            if (!ZSTD.contentEqualsIgnoreCase(contentEncodings[0])) {
                throw new RuntimeException("Only zstd content encoding is currently supported.");
            }

            // There is no Zstd decoding in Netty yet
            // so we implement this ourselves
            ZstdDecompressor decompressor = new ZstdDecompressor();

            ByteBuffer in = ByteBuffer.wrap(parameterValue);

            byte[] decompressed = new byte[1024];
            ByteBuffer nout;
            while (true) {
                decompressed = new byte[decompressed.length * 4];
                nout = ByteBuffer.wrap(decompressed);

                try {
                    decompressor.decompress(in, nout);
                    break;
                } catch (MalformedInputException ex) {
                    if (ex.getMessage().contains("Output buffer too small")) {
                        continue;
                    }
                    throw ex;
                }
            }

            parameterValue = new byte[nout.position()];
            for (int i = 0; i != nout.position(); i++) {
                parameterValue[i] = nout.array()[i];
            }
        }

        byte[] modelInputName = parameter.getName().getBytes(StandardCharsets.UTF_8);
        out.writeInt(modelInputName.length);
        out.writeBytes(modelInputName);

        encodeField(parameter.getContentType(), out);

        out.writeInt(parameterValue.length);
        out.writeBytes(parameterValue);
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
