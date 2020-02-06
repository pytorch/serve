package org.pytorch.serve.chkpnt;

import com.google.gson.JsonObject;
import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.pytorch.serve.wlm.Model;

public interface CheckpointSerializer {
    public void saveCheckpoint(
            String checkpointName,
            Map<String, Set<Entry<Double, Model>>> models,
            Map<String, String> defaultVersionsMap)
            throws IOException;

    public JsonObject getCheckpoint(String checkpointName);

    public void removeCheckpoint(String checkpointName);
}
