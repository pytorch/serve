package org.pytorch.serve.snapshot;

import java.io.IOException;
import java.util.Properties;

public interface SnapshotSerializer {

    public void saveSnapshot(Snapshot snapshot, final Properties prop) throws IOException;

    public Snapshot getSnapshot(String modelSnapshot) throws IOException;

    public Properties getLastSnapshot();
}
