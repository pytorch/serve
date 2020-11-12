package org.pytorch.serve.snapshot;

import java.io.IOException;

public interface SnapshotSerializer {

    public void saveSnapshot(Snapshot snapshot) throws IOException;

    public Snapshot getSnapshot(String modelSnapshot) throws IOException;
}
