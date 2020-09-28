package org.pytorch.serve.util;

import io.grpc.Status;

public final class GRPCUtils {

    private GRPCUtils() {}

    public static String getRegisterParam(String param, String def) {
        if ("".equals(param)) {
            return def;
        }
        return param;
    }

    public static int getRegisterParam(int param, int def) {
        if (param > 0) {
            return param;
        }
        return def;
    }

    public static Status getGRPCStatusCode(int httpStatusCode) {
        switch (httpStatusCode) {
            case 400:
                return Status.INVALID_ARGUMENT;
            case 401:
                return Status.UNAUTHENTICATED;
            case 403:
                return Status.PERMISSION_DENIED;
            case 404:
                return Status.NOT_FOUND;
            case 409:
                return Status.ABORTED;
            case 413:
            case 429:
                return Status.RESOURCE_EXHAUSTED;
            case 416:
                return Status.OUT_OF_RANGE;
            case 499:
                return Status.CANCELLED;
            case 504:
                return Status.DEADLINE_EXCEEDED;
            case 501:
                return Status.UNIMPLEMENTED;
            case 503:
                return Status.UNAVAILABLE;

            default:
                {
                    if (httpStatusCode >= 200 && httpStatusCode < 300) {
                        return Status.OK;
                    }
                    if (httpStatusCode >= 400 && httpStatusCode < 500) {
                        return Status.FAILED_PRECONDITION;
                    }
                    if (httpStatusCode >= 500 && httpStatusCode < 600) {
                        return Status.INTERNAL;
                    }
                    return Status.UNKNOWN;
                }
        }
    }
}
