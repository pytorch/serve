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

    private boolean restartInProgress;

    public static void init(ConfigManager configManager) {
        snapshotManager = new SnapshotManager(configManager);
    }

    public static SnapshotManager getInstance() {
        return snapshotManager;
    }

    private SnapshotManager(ConfigManager configManager) {
        this.snapshotSerializer = new FSSnapshotSerializer();
        this.configManager = configManager;
        this.modelManager = ModelManager.getInstance();
    }

    public void saveSnapshot(String snapshotType) {
        if (configManager.isSnapshotDisabled()) {
            return;
        }

        Map<String, Model> defModels = modelManager.getDefaultModels();
        String snapshotName = getSnapshotName(snapshotType);
        Map<String, Map<String, ModelInfo>> modelNameMap = new HashMap<>();

        try {
            int modelCount = 0;
            for (Map.Entry<String, Model> m : defModels.entrySet()) {

                Set<Entry<Double, Model>> versionModels =
                        modelManager.getAllModelVersions(m.getKey());
                Map<String, ModelInfo> modelInfoMap = new HashMap<>();
                for (Entry<Double, Model> versionedModel : versionModels) {
                    ModelInfo model = new ModelInfo();
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

    public void restore(String modelSnapshot) {
        Snapshot snapshot = null;

        try {
            logger.info("Started restoring models from snapshot");
            restartInProgress = true;
            while (modelManager.isUnregisterInProgress()) {
                Thread.sleep(5000);
            }
            snapshot = snapshotSerializer.getSnapshot(modelSnapshot);
            // Validate snapshot
            validate(snapshot);
            // Init. models
            initModels(snapshot);
        } catch (InvalidSnapshotException e) {
            logger.error("Error while validating snapshot. Details: {}", e.getCause());
        } catch (IOException e) {
            logger.error("Error loading snapshot {}", snapshot.getName());
        } catch (InterruptedException e) {
            logger.error("Error encountered while loading snapshot");
            logger.error(e.getMessage());
        } finally {
            restartInProgress = false;
        }
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

            Map<String, Map<String, ModelInfo>> models = snapshot.getModels();

            for (Map.Entry<String, Map<String, ModelInfo>> modelMap : models.entrySet()) {
                String modelName = modelMap.getKey();
                String defVersionId = null;
                for (Map.Entry<String, ModelInfo> versionModel : modelMap.getValue().entrySet()) {
                    String versionId = versionModel.getKey();
                    ModelInfo modelInfo = versionModel.getValue();
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

        Map<String, Map<String, ModelInfo>> models = snapshot.getModels();
        for (Map.Entry<String, Map<String, ModelInfo>> modelMap : models.entrySet()) {
            String modelName = modelMap.getKey();
            for (Map.Entry<String, ModelInfo> versionModel : modelMap.getValue().entrySet()) {
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

    public boolean isRestartInProgress() {
        return restartInProgress;
    }

    private String getSnapshotName(String snapshotType) {
        return new SimpleDateFormat("yyyyMMddHHmmss'-" + snapshotType + ".cfg'").format(new Date());
    }
}
