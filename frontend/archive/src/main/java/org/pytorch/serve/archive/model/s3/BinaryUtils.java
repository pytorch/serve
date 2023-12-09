package org.pytorch.serve.archive.s3;

import java.util.Locale;

/** Utilities for encoding and decoding binary data to and from different forms. */
public final class BinaryUtils {
    private BinaryUtils() {}

    /**
     * Converts byte data to a Hex-encoded string.
     *
     * @param data data to hex encode.
     * @return hex-encoded string.
     */
    public static String toHex(byte[] data) {
        StringBuilder sb = new StringBuilder(data.length * 2);
        for (byte b : data) {
            String hex = Integer.toHexString(b);
            if (hex.length() == 1) {
                // Append leading zero.
                sb.append('0');
            } else if (hex.length() == 8) {
                // Remove ff prefix from negative numbers.
                hex = hex.substring(6);
            }
            sb.append(hex);
        }
        return sb.toString().toLowerCase(Locale.getDefault());
    }

    /**
     * Converts a Hex-encoded data string to the original byte data.
     *
     * @param hexData hex-encoded data to decode.
     * @return decoded data from the hex string.
     */
    public static byte[] fromHex(String hexData) {
        byte[] result = new byte[(hexData.length() + 1) / 2];
        String hexNumber;
        int stringOffset = 0;
        int byteOffset = 0;
        while (stringOffset < hexData.length()) {
            hexNumber = hexData.substring(stringOffset, stringOffset + 2);
            stringOffset += 2;
            result[byteOffset++] = (byte) Integer.parseInt(hexNumber, 16);
        }
        return result;
    }
}
