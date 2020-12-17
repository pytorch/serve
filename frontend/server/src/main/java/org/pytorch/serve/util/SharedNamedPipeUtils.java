package org.pytorch.serve.util;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public final class SharedNamedPipeUtils {

    private SharedNamedPipeUtils() {}

    public static String getSharedNamedPipePath(String port) {
        return System.getProperty("java.io.tmpdir") + "worker_" + port;
    }

    public static String getSharedNamedPipeStdOut(String port) {
        return getSharedNamedPipePath(port) + ".out";
    }

    public static String getSharedNamedPipeStdErr(String port) {
        return getSharedNamedPipePath(port) + ".err";
    }

    public static void cleanupSharedNamedPipePathFiles(String port) throws IOException {
        Files.deleteIfExists(new File(getSharedNamedPipeStdOut(port)).toPath());
        Files.deleteIfExists(new File(getSharedNamedPipeStdErr(port)).toPath());
    }
}
