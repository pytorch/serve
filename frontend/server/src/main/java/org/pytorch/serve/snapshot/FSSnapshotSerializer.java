package org.pytorch.serve.snapshot;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.http.ConflictStatusException;
import org.pytorch.serve.util.ConfigManager;

public class FSSnapshotSerializer implements SnapshotSerializer {

    private ConfigManager configManager = ConfigManager.getInstance();
    private static final String TS_MODEL_SNAPSHOT = "model_snapshot";
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public void saveSnapshot(Snapshot snapshot) throws IOException, ConflictStatusException {
        File snapshotPath = new File(System.getProperty("LOG_LOCATION") + "/config");

        FileUtils.forceMkdir(snapshotPath);

        Properties prop = configManager.getConfiguration();

        File snapshotFile = new File(snapshotPath, snapshot.getName());
        if (snapshotFile.exists()) {
            throw new ConflictStatusException(
                    "Snapshot " + snapshot.getName() + " already exists.");
        }

        String snapshotJson = GSON.toJson(snapshot, Snapshot.class);
        prop.put(TS_MODEL_SNAPSHOT, snapshotJson);
        try (OutputStream os = new FileOutputStream(snapshotFile)) {
            OutputStreamWriter osWriter = new OutputStreamWriter(os, StandardCharsets.UTF_8);
            prop.store(osWriter, "Saving snapshot");
            osWriter.flush();
            osWriter.close();
        }
    }

    public Snapshot getSnapshot(String snapshotJson) throws IOException {
        return GSON.fromJson(snapshotJson, Snapshot.class);
    }

    public List<Snapshot> getAllSnapshots() throws IOException {
        ArrayList<Snapshot> resp = new ArrayList<Snapshot>();
        String[] snapshots = new File(System.getProperty("LOG_LOCATION") + "/config").list();
        if (snapshots != null) {
            for (String snapshotName : snapshots) {
                resp.add(getSnapshot(snapshotName));
            }
        }
        return resp;
    }

    public void removeSnapshot(String snapshotName) throws IOException {
        String snapshotPath = getSnapshotPath(snapshotName);
        FileUtils.deleteDirectory(new File(snapshotPath));
    }

    private String getSnapshotPath(String snapshotName) {
        return System.getProperty("LOG_LOCATION") + "/config" + "/" + snapshotName;
    }
}
