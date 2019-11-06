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
package com.amazonaws.ml.mms.archive;

public final class Hex {

    private static final char[] HEX_CHARS = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    };

    private Hex() {}

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
