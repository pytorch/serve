package org.pytorch.serve.chkpnt;

import com.google.gson.JsonObject;
import java.io.IOException;
import java.util.Map;

public interface CheckpointSerializer {

    public void saveCheckpoint(Checkpoint chkpnt, Map<String, String> versionMarPath)
            throws IOException;

    public JsonObject getCheckpoint(String checkpointName);

    public void removeCheckpoint(String checkpointName);
}
