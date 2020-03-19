package org.pytorch.serve.util.codec;

import io.netty.buffer.ByteBuf;
import io.netty.handler.codec.CorruptedFrameException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public final class CodecUtils {

    public static final int END = -1;
    public static final int BUFFER_UNDER_RUN = -3;

    private CodecUtils() {}

    public static int readLength(ByteBuf byteBuf, int maxLength) {
        int size = byteBuf.readableBytes();
        if (size < 4) {
            return BUFFER_UNDER_RUN;
        }

        int len = byteBuf.readInt();
        if (len > maxLength) {
            throw new CorruptedFrameException("Message size exceed limit: " + len);
        }
        if (len > byteBuf.readableBytes()) {
            return BUFFER_UNDER_RUN;
        }
        return len;
    }

    public static String readString(ByteBuf byteBuf, int len) {
        return new String(read(byteBuf, len), StandardCharsets.UTF_8);
    }

    public static byte[] read(ByteBuf in, int len) {
        if (len < 0) {
            throw new CorruptedFrameException("Invalid message size: " + len);
        }

        byte[] buf = new byte[len];
        in.readBytes(buf);
        return buf;
    }

    public static Map<String, String> readMap(ByteBuf in, int len) {
        HashMap<String, String> ret = new HashMap<>();
        for (; len > 0; len--) {
            int l = readLength(in, in.readableBytes());
            String key = readString(in, l);
            l = readLength(in, in.readableBytes());
            String val = readString(in, l);
            ret.put(key, val);
        }
        return ret;
    }
}
