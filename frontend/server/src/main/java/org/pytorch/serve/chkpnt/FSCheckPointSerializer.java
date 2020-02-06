package org.pytorch.serve.chkpnt;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;

public class FSCheckPointSerializer implements CheckpointSerializer {

    private ConfigManager configManager = ConfigManager.getInstance();

    public void saveCheckpoint(
            String checkpointName,
            Map<String, Set<Entry<Double, Model>>> models,
            Map<String, String> defaultVersionsMap)
            throws IOException {

        long created = System.currentTimeMillis();
        JsonObject modelCheckpoint = new JsonObject();
        modelCheckpoint.addProperty("created", created);
        for (Map.Entry<String, Set<Entry<Double, Model>>> model : models.entrySet()) {
            for (Map.Entry<Double, Model> versionedModels : model.getValue()) {
                Model vmodel = versionedModels.getValue();
                JsonObject modelData = new JsonObject();
                JsonObject modelVersionData = new JsonObject();
                modelVersionData.addProperty(
                        "default",
                        vmodel.getVersion().equals(defaultVersionsMap.get(model.getKey())));
                modelVersionData.addProperty("marName", vmodel.getModelUrl());
                modelVersionData.addProperty("minWorkers", vmodel.getMinWorkers());
                modelVersionData.addProperty("maxWorkers", vmodel.getMaxWorkers());
                modelVersionData.addProperty("batchSize", vmodel.getBatchSize());
                modelVersionData.addProperty("maxBatchDelay", vmodel.getMaxBatchDelay());
                // modelVersionData.addProperty("path", vmodel.getModelDir().getAbsolutePath());
                // TODO add logic to move mar to checkpoint directory.
                modelData.add(String.valueOf(versionedModels.getKey()), modelVersionData);
                modelCheckpoint.add(model.getKey(), modelData);
                try (FileWriter file =
                        new FileWriter(
                                configManager.getCheckpointStore()
                                        + "/"
                                        + checkpointName
                                        + ".json")) {
                    file.write(modelCheckpoint.toString());
                    file.flush();
                }
            }
        }
    }

    public JsonObject getCheckpoint(String checkpointName) {
        JsonParser jsonParser = new JsonParser();

        JsonObject checkpointJson = null;
        try (FileReader reader =
                new FileReader(
                        configManager.getCheckpointStore() + "/" + checkpointName + ".json")) {
            checkpointJson = jsonParser.parse(reader).getAsJsonObject();

        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return checkpointJson;
    }

    public void removeCheckpoint(String checkpointName) {}
}
