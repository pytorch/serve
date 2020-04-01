package org.pytorch.serve.snapshot;

import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.apache.commons.io.FilenameUtils;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class SnapshotManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

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
        Map<String, Map<String, ModelSnapshot>> modelNameMap = new HashMap<>();

        try {
            int modelCount = 0;
            for (Map.Entry<String, Model> m : defModels.entrySet()) {

                Set<Entry<Double, Model>> versionModels =
                        modelManager.getAllModelVersions(m.getKey());
                Map<String, ModelSnapshot> modelInfoMap = new HashMap<>();
                for (Entry<Double, Model> versionedModel : versionModels) {
                    ModelSnapshot model = new ModelSnapshot();
                    String version = String.valueOf(versionedModel.getKey());
                    model.setBatchSize(versionedModel.getValue().getBatchSize());
                    model.setDefaultVersion(
                            m.getValue()
                                    .getVersion()
                                    .equals(versionedModel.getValue().getVersion()));
                    model.setMarName(
                            FilenameUtils.getName(versionedModel.getValue().getModelUrl()));
                    model.setMaxBatchDelay(versionedModel.getValue().getMaxBatchDelay());
                    model.setMaxWorkers(versionedModel.getValue().getMaxWorkers());
                    model.setMinWorkers(versionedModel.getValue().getMinWorkers());
                    model.setResponseTimeout(versionedModel.getValue().getResponseTimeout());
                    modelInfoMap.put(version, model);
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
    public List<Snapshot> getSnapshots() throws SnapshotReadException {
        try {
            return snapshotSerializer.getAllSnapshots();
        } catch (IOException e) {
            throw new SnapshotReadException(
                    "Error while retrieving snapshot details. Cause : " + e.getCause());
        }
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

    private void terminateModels() throws ModelNotFoundException {
        Map<String, Model> defModels = modelManager.getDefaultModels();

        for (Map.Entry<String, Model> m : defModels.entrySet()) {

            Set<Entry<Double, Model>> versionModels = modelManager.getAllModelVersions(m.getKey());
            String defaultVersionId = m.getValue().getVersion();
            for (Entry<Double, Model> versionedModel : versionModels) {
                String versionId = String.valueOf(versionedModel.getKey());
                if (defaultVersionId.equals(versionId)) {
                    continue;
                }
                modelManager.unregisterModel(versionedModel.getValue().getModelName(), versionId);
            }
            modelManager.unregisterModel(m.getValue().getModelName(), defaultVersionId);
        }
    }

    private void initModels(Snapshot snapshot) {
        try {

            Map<String, Map<String, ModelSnapshot>> models = snapshot.getModels();

            for (Map.Entry<String, Map<String, ModelSnapshot>> modelMap : models.entrySet()) {
                String modelName = modelMap.getKey();
                String defVersionId = null;
                for (Map.Entry<String, ModelSnapshot> versionModel :
                        modelMap.getValue().entrySet()) {
                    String versionId = versionModel.getKey();
                    ModelSnapshot modelInfo = versionModel.getValue();
                    // TODO init/register models
                    modelManager.registerModel(
                            modelInfo.getMarName(),
                            modelName,
                            null,
                            null,
                            modelInfo.getBatchSize(),
                            modelInfo.getMaxBatchDelay(),
                            modelInfo.getResponseTimeout(),
                            modelName);
                    modelManager.updateModel(
                            modelName,
                            versionId,
                            modelInfo.getMinWorkers(),
                            modelInfo.getMaxWorkers(),
                            true);
                    if (modelInfo.getDefaultVersion()) {
                        defVersionId = versionId;
                    }
                }
                modelManager.setDefaultVersion(modelName, defVersionId);
            }

        } catch (IOException e) {
            logger.error("Error while retrieving snapshot details. Details: {}", e.getMessage());
        } catch (ModelException e) {
            logger.error("Error while registering model. Details: {}", e.getMessage());
        }
    }

    public HttpResponseStatus removeSnapshot(String snapshotName) {
        HttpResponseStatus httpResponseStatus = HttpResponseStatus.OK;
        try {
            snapshotSerializer.removeSnapshot(snapshotName);
        } catch (IOException e) {
            httpResponseStatus = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }
        return httpResponseStatus;
    }

    private boolean validate(Snapshot snapshot) throws IOException, InvalidSnapshotException {
        logger.info("Validating snapshot {}", snapshot.getName());
        String modelStore = configManager.getModelStore();

        Map<String, Map<String, ModelSnapshot>> models = snapshot.getModels();
        for (Map.Entry<String, Map<String, ModelSnapshot>> modelMap : models.entrySet()) {
            String modelName = modelMap.getKey();
            for (Map.Entry<String, ModelSnapshot> versionModel : modelMap.getValue().entrySet()) {
                String versionId = versionModel.getKey();
                String marName = versionModel.getValue().getMarName();
                File marFile = new File(modelStore + "/" + marName);
                if (!marFile.exists()) {
                    logger.error(
                            "Correspoding mar file for model {}, version {} not found in model store",
                            modelName,
                            versionId);
                    throw new InvalidSnapshotException(
                            "Correspoding mar file for model :"
                                    + modelName
                                    + ", version :"
                                    + versionId
                                    + " not found in model store");
                }
            }
        }
        logger.info("Validated snapshot {}", snapshot.getName());
        return true;
    }

    public void getLastSnapshot() throws IOException {
        // TODO Auto-generated method stub

    }

    private String getSnapshotName(String snapshotType) {
        return new SimpleDateFormat("yyyyMMddHHmmssSSS'-" + snapshotType + ".cfg'")
                .format(new Date());
    }
}
