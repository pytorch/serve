package org.pytorch.serve.util;

public final class SharedNamedPipeUtils {

    public static final String getSharedNamedPipePath(String port) {
        return System.getProperty("java.io.tmpdir") + "worker_" + port;
    }

    public static final String getSharedNamedPipeStdOut(String port) {
        return getSharedNamedPipePath(port) + ".out";
    }

    public static final String getSharedNamedPipeStdErr(String port) {
        return getSharedNamedPipePath(port) + ".err";
    }

    private SharedNamedPipeUtils() {}
}
