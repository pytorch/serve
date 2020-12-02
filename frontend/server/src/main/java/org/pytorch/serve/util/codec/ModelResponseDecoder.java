package org.pytorch.serve.util.codec;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageDecoder;
import java.util.ArrayList;
import java.util.List;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.Predictions;

public class ModelResponseDecoder extends ByteToMessageDecoder {

    private final int maxBufferSize;

    public ModelResponseDecoder(int maxBufferSize) {
        this.maxBufferSize = maxBufferSize;
    }

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
        int size = in.readableBytes();
        if (size < 9) {
            return;
        }

        in.markReaderIndex();
        boolean completed = false;
        try {
            ModelWorkerResponse resp = new ModelWorkerResponse();
            // Get Response overall Code
            resp.setCode(in.readInt());

            int len = CodecUtils.readLength(in, maxBufferSize);
            if (len == CodecUtils.BUFFER_UNDER_RUN) {
                return;
            }
            resp.setMessage(CodecUtils.readString(in, len));

            List<Predictions> predictions = new ArrayList<>();
            while ((len = CodecUtils.readLength(in, maxBufferSize)) != CodecUtils.END) {
                if (len == CodecUtils.BUFFER_UNDER_RUN) {
                    return;
                }
                Predictions prediction = new Predictions();
                // Set response RequestId
                prediction.setRequestId(CodecUtils.readString(in, len));

                len = CodecUtils.readLength(in, maxBufferSize);
                if (len == CodecUtils.BUFFER_UNDER_RUN) {
                    return;
                }
                // Set content type
                prediction.setContentType(CodecUtils.readString(in, len));

                // Set per request response code
                int httpStatusCode = in.readInt();
                prediction.setStatusCode(httpStatusCode);

                // Set the actual message
                len = CodecUtils.readLength(in, maxBufferSize);
                if (len == CodecUtils.BUFFER_UNDER_RUN) {
                    return;
                }
                prediction.setReasonPhrase(CodecUtils.readString(in, len));

                len = CodecUtils.readLength(in, maxBufferSize);
                if (len == CodecUtils.BUFFER_UNDER_RUN) {
                    return;
                }
                prediction.setHeaders(CodecUtils.readMap(in, len));

                len = CodecUtils.readLength(in, maxBufferSize);
                if (len == CodecUtils.BUFFER_UNDER_RUN) {
                    return;
                }
                prediction.setResp(CodecUtils.read(in, len));
                predictions.add(prediction);
            }
            resp.setPredictions(predictions);
            out.add(resp);
            completed = true;
        } finally {
            if (!completed) {
                in.resetReaderIndex();
            }
        }
    }
}
