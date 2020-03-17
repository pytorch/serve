package org.pytorch.serve.archive;

public final class HexUtils {

    private static final char[] HEX_CHARS = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    };

    private HexUtils() {}

    public static String toHexString(byte[] block) {
        return toHexString(block, 0, block.length);
    }

    public static String toHexString(byte[] block, int offset, int len) {
        if (block == null) {
            return null;
        }
        if (offset < 0 || offset + len > block.length) {
            throw new IllegalArgumentException("Invalid offset or length.");
        }

        StringBuilder buf = new StringBuilder();
        for (int i = offset, size = offset + len; i < size; i++) {
            int high = (block[i] & 0xf0) >> 4;
            int low = block[i] & 0x0f;

            buf.append(HEX_CHARS[high]);
            buf.append(HEX_CHARS[low]);
        }
        return buf.toString();
    }
}
