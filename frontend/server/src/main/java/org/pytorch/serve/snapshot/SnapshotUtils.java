package org.pytorch.serve.snapshot;

public class SnapshotUtils {

    public static String getLastSnapshot(String storageType) {
        if (storageType == "FS") {
            return FSSnapshotSerializer.getLastSnapshotFS();
        }
        return null;
    }
}
