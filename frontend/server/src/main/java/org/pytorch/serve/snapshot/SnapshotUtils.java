package org.pytorch.serve.snapshot;

import lombok.experimental.UtilityClass;

@UtilityClass
public final class SnapshotUtils {

    public static String getLastSnapshot(String storageType) {
        if ("FS".equalsIgnoreCase(storageType)) {
            return FSSnapshotSerializer.getLastSnapshotFS();
        }
        return null;
    }
}
