package org.pytorch.serve.snapshot;

import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.servingsdk.snapshot.SnapshotSerializer;

public final class SnapshotSerializerFactory {
    private static SnapshotSerializer snapshotSerializer;

    private SnapshotSerializerFactory() {}

    private static synchronized void initialize() {
        snapshotSerializer = PluginsManager.getInstance().getSnapShotSerializer();
        if (snapshotSerializer == null) {
            snapshotSerializer = new FSSnapshotSerializer();
        }
    }

    public static synchronized SnapshotSerializer getSerializer() {
        if (snapshotSerializer == null) {
            SnapshotSerializerFactory.initialize();
        }
        return snapshotSerializer;
    }
}
