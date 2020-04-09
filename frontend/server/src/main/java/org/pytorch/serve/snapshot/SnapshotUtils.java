package org.pytorch.serve.snapshot;

public final class SnapshotUtils {

    private SnapshotUtils() {}

    public static String getLastSnapshot(String storageType) {
        if ("FS".equalsIgnoreCase(storageType)) {
            return FSSnapshotSerializer.getLastSnapshotFS();
        }
        return null;
    }
}
