package org.pytorch.serve.snapshot;

import java.util.Properties;
import org.pytorch.serve.snapshot.ext.aws.DDBSnapshotSerializer;

public final class SnapshotUtils {

    private SnapshotUtils() {}

    public static Properties getLastSnapshot(String storageType) {
        if ("FS".equalsIgnoreCase(storageType)) {
            return FSSnapshotSerializer.getLastSnapshotFS();
        } else if ("DDB".equalsIgnoreCase(storageType)) {
            return DDBSnapshotSerializer.getLastSnapshot();
        }
        return null;
    }
}
