package org.pytorch.serve.snapshot;

import java.io.IOException;
import java.util.List;

public interface SnapshotSerializer {

    public void saveSnapshot(Snapshot snapshot) throws IOException;

    public Snapshot getSnapshot(String modelSnapshot) throws IOException;

    public List<Snapshot> getAllSnapshots() throws IOException;

    public void removeSnapshot(String snapshotName) throws IOException;
}
