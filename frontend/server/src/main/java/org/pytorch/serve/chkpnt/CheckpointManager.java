package org.pytorch.serve.chkpnt;

import io.netty.handler.codec.http.HttpResponseStatus;
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
        Map<String, Set<Entry<Double, Model>>> modelMap =
                new HashMap<String, Set<Entry<Double, Model>>>();
        for (Map.Entry<String, Model> m : defModels.entrySet()) {
            try {
                Set<Entry<Double, Model>> models = modelMgr.getAllModelVersions(m.getKey());
                modelMap.put(m.getKey(), models);

            } catch (ModelNotFoundException e) {
                logger.error("Model not found while saving checkpoint {}", chkpntName);
                response = HttpResponseStatus.INTERNAL_SERVER_ERROR;
            }
        }
        chkpntSerializer.saveCheckpoint(chkpntName, modelMap);
        return response;
    }

    public List<String> getCheckpoints(String chkpntName) {

        return null;
    }

    public HttpResponseStatus restartwithCheckpoint(String chkpntName) {
        String chkpntStore = configManager.getCheckpointStore();

        return null;
    }

    public HttpResponseStatus removeCheckpoint(String chkpntName) {

        return null;
    }
}
