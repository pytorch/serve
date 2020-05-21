package org.pytorch.serve.snapshot;

import com.google.gson.JsonObject;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class SnapshotManager {

    private static final Logger logger = LoggerFactory.getLogger(SnapshotManager.class);

    private static SnapshotManager snapshotManager;

    private ConfigManager configManager;
    private ModelManager modelManager;
    private SnapshotSerializer snapshotSerializer;

    public static void init(ConfigManager configManager) {
        snapshotManager = new SnapshotManager(configManager);
    }

    public static SnapshotManager getInstance() {
        return snapshotManager;
    }

    private SnapshotManager(ConfigManager configManager) {
        this.configManager = configManager;
        this.modelManager = ModelManager.getInstance();
        this.snapshotSerializer =
                SnapshotSerializerFactory.getSerializer(configManager.getSnapshotStore());
    }

    private void saveSnapshot(String snapshotName) {
        if (configManager.isSnapshotDisabled()) {
            return;
        }

        Map<String, Model> defModels = modelManager.getDefaultModels();
        Map<String, Map<String, JsonObject>> modelNameMap = new HashMap<>();

        try {
            int modelCount = 0;
            for (Map.Entry<String, Model> m : defModels.entrySet()) {

                Set<Entry<String, Model>> versionModels =
                        modelManager.getAllModelVersions(m.getKey());

                Map<String, JsonObject> modelInfoMap = new HashMap<>();
                for (Entry<String, Model> versionedModel : versionModels) {
                    String version = String.valueOf(versionedModel.getKey());
                    boolean isDefaultVersion =
                            m.getValue()
                                    .getVersion()
                                    .equals(versionedModel.getValue().getVersion());

                    modelInfoMap.put(
                            version, versionedModel.getValue().getModelState(isDefaultVersion));
                    ++modelCount;
                }
                modelNameMap.put(m.getKey(), modelInfoMap);
            }

            Snapshot snapshot = new Snapshot(snapshotName, modelCount);
            snapshot.setModels(modelNameMap);
            snapshotSerializer.saveSnapshot(snapshot);
        } catch (ModelNotFoundException e) {
            logger.error("Model not found while saving snapshot {}", snapshotName);
        } catch (IOException e) {
            logger.error("Error while saving snapshot to file {}", snapshotName);
        }
    }

    public void saveSnapshot() {
        saveSnapshot(getSnapshotName("snapshot"));
    }

    public void saveStartupSnapshot() {
        saveSnapshot(getSnapshotName("startup"));
    }

    public void saveShutdownSnapshot() {
        saveSnapshot(getSnapshotName("shutdown"));
    }

    @SuppressWarnings("PMD")
    public Snapshot getSnapshot(String snapshotName) throws SnapshotReadException {
        try {
            return snapshotSerializer.getSnapshot(snapshotName);
        } catch (IOException e) {
            throw new SnapshotReadException(
                    "Error while retrieving snapshot details. Cause : " + e.getCause());
        }
    }

    public void restore(String modelSnapshot) throws InvalidSnapshotException, IOException {
        Snapshot snapshot = null;

        logger.info("Started restoring models from snapshot {}", modelSnapshot);
        snapshot = snapshotSerializer.getSnapshot(modelSnapshot);
        // Validate snapshot
        validate(snapshot);
        // Init. models
        initModels(snapshot);
    }

    private void initModels(Snapshot snapshot) {
        try {

            Map<String, Map<String, JsonObject>> models = snapshot.getModels();

            if (snapshot.getModelCount() <= 0) {
                logger.warn("Model snapshot is empty. Starting TorchServe without initial models.");
                return;
            }

            for (Map.Entry<String, Map<String, JsonObject>> modelMap : models.entrySet()) {
                String modelName = modelMap.getKey();
                for (Map.Entry<String, JsonObject> versionModel : modelMap.getValue().entrySet()) {
                    JsonObject modelInfo = versionModel.getValue();
                    modelManager.registerAndUpdateModel(modelName, modelInfo);
                }
            }

        } catch (IOException e) {
            logger.error("Error while retrieving snapshot details. Details: {}", e.getMessage());
        } catch (ModelException e) {
            logger.error("Error while registering model. Details: {}", e.getMessage());
        }
    }

    private boolean validate(Snapshot snapshot) throws IOException, InvalidSnapshotException {
        logger.info("Validating snapshot {}", snapshot.getName());
        String modelStore = configManager.getModelStore();

        Map<String, Map<String, JsonObject>> models = snapshot.getModels();
        for (Map.Entry<String, Map<String, JsonObject>> modelMap : models.entrySet()) {
            String modelName = modelMap.getKey();
            for (Map.Entry<String, JsonObject> versionModel : modelMap.getValue().entrySet()) {
                String versionId = versionModel.getKey();
                String marName = versionModel.getValue().get(Model.MAR_NAME).getAsString();
                File marFile = new File(modelStore + "/" + marName);
                if (!marFile.exists()) {
                    logger.error(
                            "Model archive file for model {}, version {} not found in model store",
                            modelName,
                            versionId);
                    throw new InvalidSnapshotException(
                            "Model archive file for model :"
                                    + modelName
                                    + ", version :"
                                    + versionId
                                    + " not found in model store");
                }
            }
        }
        logger.info("Snapshot {} validated successfully", snapshot.getName());
        return true;
    }

    private String getSnapshotName(String snapshotType) {
        return new SimpleDateFormat("yyyyMMddHHmmssSSS'-" + snapshotType + ".cfg'")
                .format(new Date());
    }
}
