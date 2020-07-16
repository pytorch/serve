package org.pytorch.serve.snapshot;

import org.pytorch.serve.snapshot.ext.aws.DDBSnapshotSerializer;

public final class SnapshotSerializerFactory {
    private static SnapshotSerializer snapshotSerializer;

    private SnapshotSerializerFactory() {}

    private static synchronized void initialize(String storageType) {
        if (snapshotSerializer == null) {
            if ("FS".equalsIgnoreCase(storageType)) {
                snapshotSerializer = new FSSnapshotSerializer();
            }
            if ("DDB".equalsIgnoreCase(storageType)) {
                snapshotSerializer = new DDBSnapshotSerializer();
            }
        }
    }

    public static synchronized SnapshotSerializer getSerializer(String storageType) {
        if (snapshotSerializer == null) {
            SnapshotSerializerFactory.initialize(storageType);
        }
        return snapshotSerializer;
    }
}
