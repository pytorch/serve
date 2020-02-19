package org.pytorch.serve.checkpoint;

import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class CheckpointManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static CheckpointManager chkpntManager;

    private ConfigManager configManager;
    private CheckpointSerializer chkpntSerializer;

    private boolean restartInProgress;

    public static void init(ConfigManager configManager) {
        chkpntManager = new CheckpointManager(configManager);
    }

    public static CheckpointManager getInstance() {
        return chkpntManager;
    }

    private CheckpointManager(ConfigManager configManager) {
        // TODO - Serialize init. can move to ModelServer or it can be initialized based on config
        this.chkpntSerializer = new FSCheckpointSerializer();
        this.configManager = configManager;
    }

    public HttpResponseStatus saveCheckpoint(String chkpntName) {
        HttpResponseStatus response = HttpResponseStatus.OK;
        ModelManager modelMgr = ModelManager.getInstance();
        Map<String, Model> defModels = modelMgr.getDefaultModels();
        Map<String, String> versionMarPath = new HashMap<String, String>();

        Map<String, Map<String, ModelInfo>> modelNameMap = new HashMap<>();

        try {
            int modelCount = 0;
            for (Map.Entry<String, Model> m : defModels.entrySet()) {

                Set<Entry<Double, Model>> versionModels = modelMgr.getAllModelVersions(m.getKey());
                Map<String, ModelInfo> modelInfoMap = new HashMap<>();
                for (Entry<Double, Model> versionedModel : versionModels) {
                    ModelInfo model = new ModelInfo();
                    String version = String.valueOf(versionedModel.getKey());
                    model.setBatchSize(versionedModel.getValue().getBatchSize());
                    model.setDefaultVersion(
                            m.getValue()
                                    .getVersion()
                                    .equals(versionedModel.getValue().getVersion()));
                    model.setMarName(m.getKey() + "_" + version);
                    model.setMaxBatchDelay(versionedModel.getValue().getMaxBatchDelay());
                    model.setMaxWorkers(versionedModel.getValue().getMaxWorkers());
                    model.setMinWorkers(versionedModel.getValue().getMinWorkers());
                    model.setResponseTimeout(versionedModel.getValue().getResponseTimeout());
                    modelInfoMap.put(version, model);
                    versionMarPath.put(
                            m.getKey() + "_" + versionedModel.getValue().getVersion(),
                            versionedModel.getValue().getModelDir().getAbsolutePath());
                    ++modelCount;
                }
                modelNameMap.put(m.getKey(), modelInfoMap);
            }

            Checkpoint checkpoint = new Checkpoint(chkpntName, modelCount);
            checkpoint.setModels(modelNameMap);
            chkpntSerializer.saveCheckpoint(checkpoint, versionMarPath);
        } catch (ModelNotFoundException e) {
            logger.error("Model not found while saving checkpoint {}", chkpntName);
            response = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        } catch (IOException e) {
            logger.error("Error while saving checkpoint to file {}", chkpntName);
            response = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }

        return response;
    }

    @SuppressWarnings("PMD")
    public List<Checkpoint> getCheckpoints() throws CheckpointReadException {
        try {
            return chkpntSerializer.getAllCheckpoints();
        } catch (IOException e) {
            throw new CheckpointReadException(
                    "Error while retrieving checkpoint details. Cause : " + e.getCause());
        }
    }

    @SuppressWarnings("PMD")
    public Checkpoint getCheckpoint(String chkpntName) throws CheckpointReadException {
        try {
            return chkpntSerializer.getCheckpoint(chkpntName);
        } catch (IOException e) {
            throw new CheckpointReadException(
                    "Error while retrieving checkpoint details. Cause : " + e.getCause());
        }
    }

    public HttpResponseStatus restart(String chkpntName) {
        HttpResponseStatus status = HttpResponseStatus.OK;

        try {
            logger.info("Started restoring checkpoint {}", chkpntName);
            restartInProgress = true;
            while (ModelManager.getInstance().isUnregisterInProgress()) {
                Thread.sleep(5000);
            }
            // Validate model
            validate(chkpntName);
            // Terminate running models
            terminateModels();
            // Init. models
            status = initModels(chkpntName);
        } catch (InvalidCheckpointException e) {
            logger.error("Error while validating checkpoint. Details: {}", e.getCause());
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        } catch (ModelNotFoundException e) {
            logger.error("Model not found while saving checkpoint {}", chkpntName);
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        } catch (IOException e) {
            logger.error("Error loading checkpoint {}", chkpntName);
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        } catch (InterruptedException e) {
            logger.error("Error encountered while loading checkpoint {}", chkpntName);
            logger.error(e.getMessage());
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        } finally {
            restartInProgress = false;
        }

        return status;
    }

    private void terminateModels() throws ModelNotFoundException {
        ModelManager modelMgr = ModelManager.getInstance();
        Map<String, Model> defModels = modelMgr.getDefaultModels();

        for (Map.Entry<String, Model> m : defModels.entrySet()) {

            Set<Entry<Double, Model>> versionModels = modelMgr.getAllModelVersions(m.getKey());
            String defaultVersionId = m.getValue().getVersion();
            for (Entry<Double, Model> versionedModel : versionModels) {
                String versionId = String.valueOf(versionedModel.getKey());
                // TODO Shall we indicate for new requests that checkpoint restart is in progress...
                if (defaultVersionId.equals(versionId)) {
                    continue;
                }
                modelMgr.unregisterModel(versionedModel.getValue().getModelName(), versionId);
            }
            modelMgr.unregisterModel(m.getValue().getModelName(), defaultVersionId);
        }
    }

    private HttpResponseStatus initModels(String chkpntName) {
        HttpResponseStatus status = HttpResponseStatus.OK;
        try {
            File chkpntMarStore =
                    new File(
                            configManager.getCheckpointStore()
                                    + "//"
                                    + chkpntName
                                    + "//model_store");
            ModelManager modelMgr = ModelManager.getInstance();
            File modelStore = new File(configManager.getModelStore());
            FileUtils.cleanDirectory(modelStore);
            FileUtils.copyDirectory(chkpntMarStore, modelStore);
            List<File> files = (List<File>) FileUtils.listFiles(modelStore, null, true);

            Checkpoint checkpoint = chkpntSerializer.getCheckpoint(chkpntName);
            Map<String, Map<String, ModelInfo>> models = checkpoint.getModels();

            if (checkpoint.getModelCount() != files.size()) {
                return HttpResponseStatus.INTERNAL_SERVER_ERROR;
            }

            for (Map.Entry<String, Map<String, ModelInfo>> modelMap : models.entrySet()) {
                String modelName = modelMap.getKey();
                String defVersionId = null;
                for (Map.Entry<String, ModelInfo> versionModel : modelMap.getValue().entrySet()) {
                    String versionId = versionModel.getKey();
                    ModelInfo modelInfo = versionModel.getValue();
                    // TODO init/register models
                    modelMgr.registerModel(
                            modelInfo.getMarName() + ".mar",
                            modelName,
                            null,
                            null,
                            modelInfo.getBatchSize(),
                            modelInfo.getMaxBatchDelay(),
                            modelInfo.getResponseTimeout(),
                            modelName);
                    CompletableFuture<HttpResponseStatus> future =
                            modelMgr.updateModel(
                                    modelName,
                                    versionId,
                                    modelInfo.getMinWorkers(),
                                    modelInfo.getMaxWorkers());
                    status = future.get();
                    if (modelInfo.getDefaultVersion()) {
                        defVersionId = versionId;
                    }
                }
                modelMgr.setDefaultVersion(modelName, defVersionId);
            }

        } catch (IOException e) {
            logger.error("Error while retrieving checkpoint details. Details: {}", e.getMessage());
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        } catch (ModelException e) {
            logger.error("Error while registering model. Details: {}", e.getMessage());
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        } catch (InterruptedException | ExecutionException e) {
            logger.error("Internal error while registering model. Details: {}", e.getMessage());
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }
        return status;
    }

    public HttpResponseStatus removeCheckpoint(String chkpntName) {
        HttpResponseStatus httpResponseStatus = HttpResponseStatus.OK;
        try {
            chkpntSerializer.removeCheckpoint(chkpntName);
        } catch (IOException e) {
            httpResponseStatus = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }
        return httpResponseStatus;
    }

    private boolean validate(String checkpointName) throws IOException, InvalidCheckpointException {
        logger.info("Validating checkpoint {}", checkpointName);
        String chkpntMarStorePath =
                configManager.getCheckpointStore() + "/" + checkpointName + "/model_store";
        Checkpoint checkpoint = chkpntSerializer.getCheckpoint(checkpointName);

        File checkPointModelStore = new File(chkpntMarStorePath);

        if (!(checkPointModelStore.exists() && checkPointModelStore.isDirectory())) {
            logger.error("Checkpoint {} does not exist", checkpointName);
            throw new InvalidCheckpointException(
                    "Checkpoint " + checkpointName + "'s model store does not exist.");
        } else {
            File[] modelsMars = checkPointModelStore.listFiles();
            if (modelsMars != null && checkpoint.getModelCount() != modelsMars.length) {
                logger.error(
                        "Model count in checkpoint {}'s model store does not match.",
                        checkpointName);
                throw new InvalidCheckpointException(
                        "Checkpoint " + checkpointName + "'s model store is corrupted.");
            }
        }

        Map<String, Map<String, ModelInfo>> models = checkpoint.getModels();
        for (Map.Entry<String, Map<String, ModelInfo>> modelMap : models.entrySet()) {
            String modelName = modelMap.getKey();
            for (Map.Entry<String, ModelInfo> versionModel : modelMap.getValue().entrySet()) {
                String versionId = versionModel.getKey();
                String marName = modelName + "_" + versionId + ".mar";
                File marFile = new File(chkpntMarStorePath + "/" + marName);
                if (!marFile.exists()) {
                    logger.error(
                            "Correspoding mar file for model {}, version {} not found in checkpoint model store",
                            checkpointName, versionId);
                    throw new InvalidCheckpointException(
                            "Correspoding mar file for model :"
                                    + modelName
                                    + ", version :"
                                    + versionId
                                    + " not found in checkpoint model store");
                }
            }
        }
        logger.info("Validated checkpoint {}", checkpointName);
        return true;
    }

    public boolean isRestartInProgress() {
        return restartInProgress;
    }
}
