package org.pytorch.serve.chkpnt;

import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkLoadManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CheckpointManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static CheckpointManager chkpntManager;

    private ConfigManager configManager;
    private WorkLoadManager wlm;
    private CheckpointSerializer chkpntSerializer;

    public static void init(ConfigManager configManager, WorkLoadManager wlm) {
        chkpntManager = new CheckpointManager(configManager, wlm);
    }

    public static CheckpointManager getInstance() {
        return chkpntManager;
    }

    private CheckpointManager(ConfigManager configManager, WorkLoadManager wlm) {
        // TODO - Serialize init. can move to ModelServer or it can be initialized based on config
        this.chkpntSerializer = new FSCheckPointSerializer();
        this.configManager = configManager;
        this.wlm = wlm;
    }

    public HttpResponseStatus saveCheckpoint(String chkpntName) {
        HttpResponseStatus response = HttpResponseStatus.OK;
        ModelManager modelMgr = ModelManager.getInstance();
        Map<String, Model> defModels = modelMgr.getDefaultModels();
        Map<String, String> versionMarPath = new HashMap<String, String>();
        Map<String, Set<Entry<Double, Model>>> modelMap =
                new HashMap<String, Set<Entry<Double, Model>>>();

        Map<String, Map<String, ModelInfo>> modelNameMap = new HashMap<>();

        try {
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
                    modelInfoMap.put(version, model);
                    versionMarPath.put(
                            m.getKey() + "_" + versionedModel.getValue().getVersion(),
                            versionedModel.getValue().getModelDir().getAbsolutePath());
                }
                modelNameMap.put(m.getKey(), modelInfoMap);
            }

            Checkpoint chkpnt = new Checkpoint(chkpntName);
            chkpnt.setModels(modelNameMap);
            chkpntSerializer.saveCheckpoint(chkpnt, versionMarPath);
        } catch (ModelNotFoundException e) {
            logger.error("Model not found while saving checkpoint {}", chkpntName);
            response = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        } catch (IOException e) {
            logger.error("Error while saving checkpoint to file {}", chkpntName);
            response = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }

        return response;
    }

    public List<Checkpoint> getCheckpoints() throws CheckpointReadException {
        try {
            return chkpntSerializer.getAllCheckpoints();
        } catch (IOException e) {
            throw new CheckpointReadException("Error while retrieving checkpoint details.");
        }
    }

    public Checkpoint getCheckpoint(String chkpntName) throws CheckpointReadException {
        try {
            return chkpntSerializer.getCheckpoint(chkpntName);
        } catch (IOException e) {
            throw new CheckpointReadException("Error while retrieving checkpoint details.");
        }
    }

    public HttpResponseStatus restartwithCheckpoint(String chkpntName) {
        Checkpoint chkpnt;
        try {
            chkpnt = chkpntSerializer.getCheckpoint(chkpntName);

            Map<String, Map<String, ModelInfo>> models = chkpnt.getModels();

            String chkpntStore = configManager.getCheckpointStore();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return null;
    }

    public HttpResponseStatus removeCheckpoint(String chkpntName) {

        return null;
    }
}
