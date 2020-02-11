package org.pytorch.serve.checkpoint;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.pytorch.serve.http.ConflictStatusException;

public interface CheckpointSerializer {

    public void saveCheckpoint(Checkpoint checkpoint, Map<String, String> versionMarPath)
            throws IOException, ConflictStatusException;

    public Checkpoint getCheckpoint(String checkpointName) throws IOException;

    public List<Checkpoint> getAllCheckpoints() throws IOException;

    public void removeCheckpoint(String checkpointName) throws IOException;
}
