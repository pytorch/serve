package org.pytorch.serve.snapshot;

import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.servingsdk.snapshot.SnapshotSerializer;

public final class SnapshotSerializerFactory {
    private static SnapshotSerializer snapshotSerializer;

    private SnapshotSerializerFactory() {}

    private static synchronized void initialize(String storageType) {
        snapshotSerializer = PluginsManager.getInstance().getSnapShotSerializer();
        if ("FS".equalsIgnoreCase(storageType)) {
            snapshotSerializer = new FSSnapshotSerializer();
        }
    }

    public static synchronized SnapshotSerializer getSerializer(String storageType) {
        if (snapshotSerializer == null) {
            SnapshotSerializerFactory.initialize(storageType);
        }
        return snapshotSerializer;
    }
}
