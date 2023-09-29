package org.pytorch.serve.util.codec;

import io.netty.buffer.ByteBuf;
import io.netty.handler.codec.TooLongFrameException;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder.NotEnoughDataDecoderException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public final class CodecUtils {

    public static final int END = -1;
    public static final int BUFFER_UNDER_RUN = -3;
    public static final long TIMEOUT_IN_MILLIS = 100;

    private CodecUtils() {}

    public static int readLength(ByteBuf byteBuf, int maxLength) {
        int size = byteBuf.readableBytes();

        if (size < 4) {
            throw new NotEnoughDataDecoderException("Did not receive enough data.");
        }

        int len = byteBuf.readInt();
        if (len > maxLength) {
            throw new TooLongFrameException(
                    "Message size exceed limit: "
                            + len
                            + "\nConsider increasing the 'max_response_size' in 'config.properties' to fix.");
        }

        if (len > byteBuf.readableBytes()) {
            throw new NotEnoughDataDecoderException("Did not receive enough data.");
        }
        return len;
    }

    public static String readString(ByteBuf byteBuf, int len) {
        return new String(read(byteBuf, len), StandardCharsets.UTF_8);
    }

    public static byte[] read(ByteBuf in, int len) {
        if (len < 0) {
            throw new NotEnoughDataDecoderException("Did not receive enough data.");
        }

        byte[] buf = new byte[len];
        in.readBytes(buf);
        return buf;
    }

    public static Map<String, String> readMap(ByteBuf in, int len) {
        HashMap<String, String> ret = new HashMap<>();
        for (; len > 0; len--) {
            int l =
                    readLength(
                            in,
                            6500000); // We replace len here with 6500000 as a workaround before we
            // can fix the whole otf. Basically, were mixing up bytes
            // (expected by readLength) and number of entries (given to
            // readMap). If we only have a small number of entries our
            // values in the map are not allowed to be very big as we
            // compare the given number of entries with the byte size
            // we're expecting after reading the length of the next
            // message.
            String key = readString(in, l);
            l = readLength(in, 6500000);
            String val = readString(in, l);
            ret.put(key, val);
        }
        return ret;
    }
}
