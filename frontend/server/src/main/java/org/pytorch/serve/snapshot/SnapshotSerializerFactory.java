package org.pytorch.serve.snapshot;

import org.pytorch.serve.snapshot.ext.aws.DDBSnapshotSerializer;

public final class SnapshotSerializerFactory {

    private SnapshotSerializerFactory() {}

    public static SnapshotSerializer getSerializer(String storageType) {
        if ("FS".equalsIgnoreCase(storageType)) {
            return new FSSnapshotSerializer();
        }
        if ("DDB".equalsIgnoreCase(storageType)) {
            return new DDBSnapshotSerializer();
        }
        return null;
    }
}
