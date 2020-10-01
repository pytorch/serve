package org.pytorch.serve.snapshot;

import org.pytorch.serve.servingsdk.snapshot.SnapshotSerializer;

public final class SnapshotSerializerFactory {

    private SnapshotSerializerFactory() {}

    public static SnapshotSerializer getSerializer(String storageType) {
        if ("FS".equalsIgnoreCase(storageType)) {
            return new FSSnapshotSerializer();
        }
        return null;
    }
}
