package org.pytorch.serve.chkpnt;

import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.pytorch.serve.wlm.Model;

public class CheckPointSerializer {

    public void saveCheckpoint(
            String checkpointName, Map<String, Set<Entry<Double, Model>>> models) {}

    public String getCheckpoint(String checkpointName) {
        return "";
    }

    public void removeCheckpoint(String checkpointName) {}
}
