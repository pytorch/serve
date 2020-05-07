package org.pytorch.serve.wlm;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import org.pytorch.serve.archive.Manifest;
import org.pytorch.serve.archive.ModelArchive;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.archive.ModelVersionNotFoundException;
import org.pytorch.serve.http.ConflictStatusException;
import org.pytorch.serve.http.InvalidModelVersionException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.NettyUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static ModelManager modelManager;

    private ConfigManager configManager;
    private WorkLoadManager wlm;
    private ConcurrentHashMap<String, ModelVersionedRefs> modelsNameMap;
    private HashSet<String> startupModels;
    private ScheduledExecutorService scheduler;

    private ModelManager(ConfigManager configManager, WorkLoadManager wlm) {
        this.configManager = configManager;
        this.wlm = wlm;
        modelsNameMap = new ConcurrentHashMap<>();
        scheduler = Executors.newScheduledThreadPool(2);
        this.startupModels = new HashSet<>();
    }

    public ScheduledExecutorService getScheduler() {
        return scheduler;
    }

    public static void init(ConfigManager configManager, WorkLoadManager wlm) {
        modelManager = new ModelManager(configManager, wlm);
    }

    public static ModelManager getInstance() {
        return modelManager;
    }

    public ModelArchive registerModel(String url, String defaultModelName)
            throws ModelException, IOException {
        return registerModel(
                url,
                null,
                null,
                null,
                1,
                100,
                configManager.getDefaultResponseTimeout(),
                defaultModelName);
    }

    public ModelArchive registerModel(
            String url,
            String modelName,
            Manifest.RuntimeType runtime,
            String handler,
            int batchSize,
            int maxBatchDelay,
            int responseTimeout,
            String defaultModelName)
            throws ModelException, IOException {

        ModelArchive archive = ModelArchive.downloadModel(configManager.getModelStore(), url);
        if (modelName == null || modelName.isEmpty()) {
            if (archive.getModelName() == null || archive.getModelName().isEmpty()) {
                archive.getManifest().getModel().setModelName(defaultModelName);
            }
        } else {
            archive.getManifest().getModel().setModelName(modelName);
        }

        String versionId = archive.getModelVersion();

        if (runtime != null) {
            archive.getManifest().setRuntime(runtime);
        }
        if (handler != null) {
            archive.getManifest().getModel().setHandler(handler);
        } else if (archive.getHandler() == null || archive.getHandler().isEmpty()) {
            archive.getManifest().getModel().setHandler(configManager.getTsDefaultServiceHandler());
        }

        archive.validate();

        Model tempModel = createModel(archive, batchSize, maxBatchDelay, responseTimeout);

        createVersionedModel(tempModel, versionId);

        logger.info("Model {} loaded.", tempModel.getModelName());

        return archive;
    }

    private Model createModel(
            ModelArchive archive, int batchSize, int maxBatchDelay, int responseTimeout) {
        Model model = new Model(archive, configManager.getJobQueueSize());
        model.setBatchSize(batchSize);
        model.setMaxBatchDelay(maxBatchDelay);
        model.setResponseTimeout(responseTimeout);

        return model;
    }

    private void createVersionedModel(Model model, String versionId)
            throws ConflictStatusException {

        ModelVersionedRefs modelVersionRef = modelsNameMap.get(model.getModelName());
        if (modelVersionRef == null) {
            modelVersionRef = new ModelVersionedRefs();
        }
        modelVersionRef.addVersionModel(model, versionId);
        modelsNameMap.putIfAbsent(model.getModelName(), modelVersionRef);
    }

    public HttpResponseStatus unregisterModel(String modelName, String versionId) {
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            logger.warn("Model not found: " + modelName);
            return HttpResponseStatus.NOT_FOUND;
        }

        if (versionId == null) {
            versionId = vmodel.getDefaultVersion();
        }

        Model model = null;
        HttpResponseStatus httpResponseStatus = HttpResponseStatus.OK;

        try {
            model = vmodel.removeVersionModel(versionId);
            model.setMinWorkers(0);
            model.setMaxWorkers(0);
            CompletableFuture<HttpResponseStatus> futureStatus = wlm.modelChanged(model, false);
            httpResponseStatus = futureStatus.get();

            // Only continue cleaning if resource cleaning succeeded

            if (httpResponseStatus == HttpResponseStatus.OK) {
                model.getModelArchive().clean();
                startupModels.remove(modelName);
                logger.info("Model {} unregistered.", modelName);
            } else {
                if (versionId == null) {
                    versionId = vmodel.getDefaultVersion();
                }
                vmodel.addVersionModel(model, versionId);
            }

            if (vmodel.getAllVersions().size() == 0) {
                modelsNameMap.remove(modelName);
            }

            ModelArchive.removeModel(configManager.getModelStore(), model.getModelUrl());
        } catch (ModelVersionNotFoundException e) {
            logger.warn("Model {} version {} not found.", modelName, versionId);
            httpResponseStatus = HttpResponseStatus.BAD_REQUEST;
        } catch (InvalidModelVersionException e) {
            logger.warn("Cannot remove default version {} for model {}", versionId, modelName);
            httpResponseStatus = HttpResponseStatus.FORBIDDEN;
        } catch (ExecutionException | InterruptedException e1) {
            logger.warn("Process was interrupted while cleaning resources.");
            httpResponseStatus = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }

        return httpResponseStatus;
    }

    public HttpResponseStatus setDefaultVersion(String modelName, String newModelVersion)
            throws InvalidModelVersionException {
        HttpResponseStatus httpResponseStatus = HttpResponseStatus.OK;
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            logger.warn("Model not found: " + modelName);
            return HttpResponseStatus.NOT_FOUND;
        }
        try {
            vmodel.setDefaultVersion(newModelVersion);
        } catch (InvalidModelVersionException e) {
            logger.warn("Model version {} does not exist for model {}", newModelVersion, modelName);
            httpResponseStatus = HttpResponseStatus.FORBIDDEN;
        }

        return httpResponseStatus;
    }

    public CompletableFuture<HttpResponseStatus> updateModel(
            String modelName, String versionId, int minWorkers, int maxWorkers, boolean isStartup) {
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            throw new AssertionError("Model not found: " + modelName);
        }

        Model model = vmodel.getVersionModel(versionId);

        if (model == null) {
            throw new AssertionError("Model version not not found for model : " + modelName);
        }

        model.setMinWorkers(minWorkers);
        model.setMaxWorkers(maxWorkers);
        logger.debug("updateModel: {}, count: {}", modelName, minWorkers);

        return wlm.modelChanged(model, isStartup);
    }

    public CompletableFuture<HttpResponseStatus> updateModel(
            String modelName, String versionId, int minWorkers, int maxWorkers) {
        return updateModel(modelName, versionId, minWorkers, maxWorkers, false);
    }

    public Map<String, Model> getDefaultModels() {
        ConcurrentHashMap<String, Model> defModelsMap = new ConcurrentHashMap<>();
        for (String key : modelsNameMap.keySet()) {
            ModelVersionedRefs mvr = modelsNameMap.get(key);
            if (mvr != null) {
                Model defaultModel = mvr.getDefaultModel();
                if (defaultModel != null) {
                    defModelsMap.put(key, defaultModel);
                }
            }
        }
        return defModelsMap;
    }

    public List<WorkerThread> getWorkers(ModelVersionName modelVersionName) {
        return wlm.getWorkers(modelVersionName);
    }

    public Map<Integer, WorkerThread> getWorkers() {
        return wlm.getWorkers();
    }

    public boolean addJob(Job job) throws ModelNotFoundException {
        String modelName = job.getModelName();
        String versionId = job.getModelVersion();
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }

        Model model = vmodel.getVersionModel(versionId);

        if (model == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }

        if (wlm.hasNoWorker(model.getModelVersionName())) {
            return false;
        }

        return model.addJob(job);
    }

    public void workerStatus(final ChannelHandlerContext ctx) {
        Runnable r =
                () -> {
                    String response = "Healthy";
                    int numWorking = 0;
                    int numScaled = 0;
                    for (Map.Entry<String, ModelVersionedRefs> m : modelsNameMap.entrySet()) {
                        numScaled += m.getValue().getDefaultModel().getMinWorkers();
                        numWorking +=
                                wlm.getNumRunningWorkers(
                                        m.getValue().getDefaultModel().getModelVersionName());
                    }

                    if ((numWorking > 0) && (numWorking < numScaled)) {
                        response = "Partial Healthy";
                    } else if ((numWorking == 0) && (numScaled > 0)) {
                        response = "Unhealthy";
                    }

                    // TODO: Check if its OK to send other 2xx errors to ALB for "Partial Healthy"
                    // and "Unhealthy"
                    NettyUtils.sendJsonResponse(
                            ctx, new StatusResponse(response), HttpResponseStatus.OK);
                };
        wlm.scheduleAsync(r);
    }

    public void modelWorkerStatus(final String modelName, final ChannelHandlerContext ctx) {
        Runnable r =
                () -> {
                    String response = "Healthy";
                    int numWorking = 0;
                    int numScaled = 0;
                    ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
                    for (Map.Entry<String, Model> m : vmodel.getAllVersions()) {
                        numScaled += m.getValue().getMinWorkers();
                        numWorking += wlm.getNumRunningWorkers(m.getValue().getModelVersionName());
                    }

                    if ((numWorking > 0) && (numWorking < numScaled)) {
                        response = "Partial Healthy";
                    } else if ((numWorking == 0) && (numScaled > 0)) {
                        response = "Unhealthy";
                    }

                    // TODO: Check if its OK to send other 2xx errors to ALB for "Partial Healthy"
                    // and "Unhealthy"
                    NettyUtils.sendJsonResponse(
                            ctx, new StatusResponse(response), HttpResponseStatus.OK);
                };
        wlm.scheduleAsync(r);
    }

    public boolean scaleRequestStatus(String modelName, String versionId) {
        Model model = modelsNameMap.get(modelName).getVersionModel(versionId);
        int numWorkers = 0;

        if (model != null) {
            numWorkers = wlm.getNumRunningWorkers(model.getModelVersionName());
        }

        return model == null || model.getMinWorkers() <= numWorkers;
    }

    public void submitTask(Runnable runnable) {
        wlm.scheduleAsync(runnable);
    }

    public Set<String> getStartupModels() {
        return startupModels;
    }

    public Model getModel(String modelName, String versionId) {
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            return null;
        }
        return vmodel.getVersionModel(versionId);
    }

    public Set<Entry<String, Model>> getAllModelVersions(String modelName)
            throws ModelNotFoundException {
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        return vmodel.getAllVersions();
    }
}
