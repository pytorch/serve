package org.pytorch.serve.chkpnt;

import java.io.IOException;
import java.util.Map;

public interface CheckpointSerializer {

    public void saveCheckpoint(Checkpoint chkpnt, Map<String, String> versionMarPath)
            throws IOException;

    public Checkpoint getCheckpoint(String checkpointName) throws IOException;

    public void removeCheckpoint(String checkpointName);
}
