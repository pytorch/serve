package org.pytorch.serve.snapshot;

public class SnapshotSerializerFactory {

    public static SnapshotSerializer getSerializer(String storageType) {
        if (storageType.equalsIgnoreCase("FS")) {
            return new FSSnapshotSerializer();
        }
        return null;
    }
}
