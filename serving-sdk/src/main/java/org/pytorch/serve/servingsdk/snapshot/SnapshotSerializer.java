package org.pytorch.serve.servingsdk.snapshot;

import java.io.IOException;
import java.util.Properties;

public interface SnapshotSerializer {

    void saveSnapshot(Snapshot snapshot, final Properties prop) throws IOException;

    Snapshot getSnapshot(String modelSnapshot) throws IOException;

    Properties getLastSnapshot();
}
