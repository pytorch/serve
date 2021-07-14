package org.pytorch.serve.wlm;

import com.google.gson.JsonObject;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.nio.file.Path;
import java.nio.file.Paths;
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
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.Manifest;
import org.pytorch.serve.archive.model.ModelArchive;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.http.ConflictStatusException;
import org.pytorch.serve.http.InvalidModelVersionException;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.messages.EnvironmentUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static ModelManager modelManager;

    private final ConfigManager configManager;
    private final WorkLoadManager wlm;
    private final ConcurrentHashMap<String, ModelVersionedRefs> modelsNameMap;
    private final HashSet<String> startupModels;
    private final ScheduledExecutorService scheduler;

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
            throws ModelException, IOException, InterruptedException, DownloadArchiveException {
        return registerModel(
                url,
                null,
                null,
                null,
                1,
                100,
                configManager.getDefaultResponseTimeout(),
                defaultModelName,
                false,
                false,
                false);
    }

    public void registerAndUpdateModel(String modelName, JsonObject modelInfo)
            throws ModelException, IOException, InterruptedException, DownloadArchiveException {

        boolean defaultVersion = modelInfo.get(Model.DEFAULT_VERSION).getAsBoolean();
        String url = modelInfo.get(Model.MAR_NAME).getAsString();

        ModelArchive archive = createModelArchive(modelName, url, null, null, modelName, false);

        Model tempModel = createModel(archive, modelInfo);

        String versionId = archive.getModelVersion();

        createVersionedModel(tempModel, versionId);

        setupModelDependencies(tempModel);
        if (defaultVersion) {
            modelManager.setDefaultVersion(modelName, versionId);
        }

        logger.info("Model {} loaded.", tempModel.getModelName());

        updateModel(modelName, versionId, true);
    }

    public ModelArchive registerModel(
            String url,
            String modelName,
            Manifest.RuntimeType runtime,
            String handler,
            int batchSize,
            int maxBatchDelay,
            int responseTimeout,
            String defaultModelName,
            boolean ignoreDuplicate,
            boolean isWorkflowModel,
            boolean s3SseKms)
            throws ModelException, IOException, InterruptedException, DownloadArchiveException {

        ModelArchive archive;
        if (isWorkflowModel && url == null) { // This is  a workflow function
            Manifest manifest = new Manifest();
            manifest.getModel().setVersion("1.0");
            manifest.getModel().setModelVersion("1.0");
            manifest.getModel().setModelName(modelName);
            manifest.getModel().setHandler(new File(handler).getName());

            File f = new File(handler.substring(0, handler.lastIndexOf(':')));
            archive = new ModelArchive(manifest, url, f.getParentFile(), true);
        } else {
            archive =
                    createModelArchive(
                            modelName, url, handler, runtime, defaultModelName, s3SseKms);
        }

        Model tempModel =
                createModel(archive, batchSize, maxBatchDelay, responseTimeout, isWorkflowModel);

        String versionId = archive.getModelVersion();

        try {
            createVersionedModel(tempModel, versionId);
        } catch (ConflictStatusException e) {
            if (!ignoreDuplicate) {
                throw e;
            }
        }

        setupModelDependencies(tempModel);

        logger.info("Model {} loaded.", tempModel.getModelName());

        return archive;
    }

    private ModelArchive createModelArchive(
            String modelName,
            String url,
            String handler,
            Manifest.RuntimeType runtime,
            String defaultModelName,
            boolean s3SseKms)
            throws ModelException, IOException, DownloadArchiveException {

        ModelArchive archive =
                ModelArchive.downloadModel(
                        configManager.getAllowedUrls(),
                        configManager.getModelStore(),
                        url,
                        s3SseKms);
        if (modelName == null || modelName.isEmpty()) {
            if (archive.getModelName() == null || archive.getModelName().isEmpty()) {
                archive.getManifest().getModel().setModelName(defaultModelName);
            }
        } else {
            archive.getManifest().getModel().setModelName(modelName);
        }

        if (runtime != null) {
            archive.getManifest().setRuntime(runtime);
        }

        if (handler != null) {
            archive.getManifest().getModel().setHandler(handler);
        } else if (archive.getHandler() == null || archive.getHandler().isEmpty()) {
            archive.getManifest().getModel().setHandler(configManager.getTsDefaultServiceHandler());
        }

        archive.getManifest().getModel().setEnvelope(configManager.getTsServiceEnvelope());

        archive.validate();

        return archive;
    }

    private void setupModelDependencies(Model model)
            throws IOException, InterruptedException, ModelException {
        String requirementsFile =
                model.getModelArchive().getManifest().getModel().getRequirementsFile();

        if (configManager.getInstallPyDepPerModel() && requirementsFile != null) {
            Path requirementsFilePath =
                    Paths.get(model.getModelDir().getAbsolutePath(), requirementsFile);

            String pythonRuntime = EnvironmentUtils.getPythonRunTime(model);

            String packageInstallCommand =
                    pythonRuntime
                            + " -m pip install -U -t "
                            + model.getModelDir().getAbsolutePath()
                            + " -r "
                            + requirementsFilePath; // NOPMD

            String[] envp =
                    EnvironmentUtils.getEnvString(
                            configManager.getModelServerHome(),
                            model.getModelDir().getAbsolutePath(),
                            null);

            Process process =
                    Runtime.getRuntime()
                            .exec(
                                    packageInstallCommand,
                                    envp,
                                    model.getModelDir().getAbsoluteFile());

            int exitCode = process.waitFor();

            if (exitCode != 0) {

                String line;
                StringBuilder outputString = new StringBuilder();
                // process's stdout is InputStream for caller process
                BufferedReader brdr =
                        new BufferedReader(new InputStreamReader(process.getInputStream()));
                while ((line = brdr.readLine()) != null) {
                    outputString.append(line);
                }
                StringBuilder errorString = new StringBuilder();
                // process's stderr is ErrorStream for caller process
                brdr = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                while ((line = brdr.readLine()) != null) {
                    errorString.append(line);
                }

                logger.info("Dependency installation stdout:\n" + outputString.toString());
                logger.error("Dependency installation stderr:\n" + errorString.toString());

                throw new ModelException(
                        "Custom pip package installation failed for " + model.getModelName());
            }
        }
    }

    private Model createModel(
            ModelArchive archive,
            int batchSize,
            int maxBatchDelay,
            int responseTimeout,
            boolean isWorkflowModel) {
        Model model = new Model(archive, configManager.getJobQueueSize());

        model.setBatchSize(
                configManager.getJsonIntValue(
                        archive.getModelName(),
                        archive.getModelVersion(),
                        Model.BATCH_SIZE,
                        batchSize));
        model.setMaxBatchDelay(
                configManager.getJsonIntValue(
                        archive.getModelName(),
                        archive.getModelVersion(),
                        Model.MAX_BATCH_DELAY,
                        maxBatchDelay));
        model.setResponseTimeout(
                configManager.getJsonIntValue(
                        archive.getModelName(),
                        archive.getModelVersion(),
                        Model.RESPONSE_TIMEOUT,
                        responseTimeout));
        model.setWorkflowModel(isWorkflowModel);

        return model;
    }

    private Model createModel(ModelArchive archive, JsonObject modelInfo) {
        Model model = new Model(archive, configManager.getJobQueueSize());
        model.setModelState(modelInfo);
        model.setWorkflowModel(false);

        return model;
    }

    private void createVersionedModel(Model model, String versionId)
            throws ModelVersionNotFoundException, ConflictStatusException {

        ModelVersionedRefs modelVersionRef = modelsNameMap.get(model.getModelName());
        if (modelVersionRef == null) {
            modelVersionRef = new ModelVersionedRefs();
        }
        modelVersionRef.addVersionModel(model, versionId);
        modelsNameMap.putIfAbsent(model.getModelName(), modelVersionRef);
    }

    public int unregisterModel(String modelName, String versionId) {
        return unregisterModel(modelName, versionId, false);
    }

    public int unregisterModel(String modelName, String versionId, boolean isCleanUp) {
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            logger.warn("Model not found: " + modelName);
            return HttpURLConnection.HTTP_NOT_FOUND;
        }

        if (versionId == null) {
            versionId = vmodel.getDefaultVersion();
        }

        Model model;
        int httpResponseStatus;

        try {
            model = vmodel.removeVersionModel(versionId);
            model.setMinWorkers(0);
            model.setMaxWorkers(0);
            CompletableFuture<Integer> futureStatus = wlm.modelChanged(model, false, isCleanUp);
            httpResponseStatus = futureStatus.get();

            // Only continue cleaning if resource cleaning succeeded

            if (httpResponseStatus == HttpURLConnection.HTTP_OK) {
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

            if (!isCleanUp && model.getModelUrl() != null) {
                ModelArchive.removeModel(configManager.getModelStore(), model.getModelUrl());
            }
        } catch (ModelVersionNotFoundException e) {
            logger.warn("Model {} version {} not found.", modelName, versionId);
            httpResponseStatus = HttpURLConnection.HTTP_BAD_REQUEST;
        } catch (InvalidModelVersionException e) {
            logger.warn("Cannot remove default version {} for model {}", versionId, modelName);
            httpResponseStatus = HttpURLConnection.HTTP_FORBIDDEN;
        } catch (ExecutionException | InterruptedException e1) {
            logger.warn("Process was interrupted while cleaning resources.");
            httpResponseStatus = HttpURLConnection.HTTP_INTERNAL_ERROR;
        }

        return httpResponseStatus;
    }

    public void setDefaultVersion(String modelName, String newModelVersion)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            logger.warn("Model not found: " + modelName);
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        vmodel.setDefaultVersion(newModelVersion);
    }

    private CompletableFuture<Integer> updateModel(
            String modelName, String versionId, boolean isStartup)
            throws ModelVersionNotFoundException {
        Model model = getVersionModel(modelName, versionId);
        return updateModel(
                modelName,
                versionId,
                model.getMinWorkers(),
                model.getMaxWorkers(),
                isStartup,
                false);
    }

    public CompletableFuture<Integer> updateModel(
            String modelName,
            String versionId,
            int minWorkers,
            int maxWorkers,
            boolean isStartup,
            boolean isCleanUp)
            throws ModelVersionNotFoundException {
        Model model = getVersionModel(modelName, versionId);

        if (model == null) {
            throw new ModelVersionNotFoundException(
                    "Model version: " + versionId + " does not exist for model: " + modelName);
        }

        model.setMinWorkers(minWorkers);
        model.setMaxWorkers(maxWorkers);
        logger.debug("updateModel: {}, count: {}", modelName, minWorkers);

        return wlm.modelChanged(model, isStartup, isCleanUp);
    }

    private Model getVersionModel(String modelName, String versionId) {
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            throw new AssertionError("Model not found: " + modelName);
        }

        return vmodel.getVersionModel(versionId);
    }

    public CompletableFuture<Integer> updateModel(
            String modelName, String versionId, int minWorkers, int maxWorkers)
            throws ModelVersionNotFoundException {
        return updateModel(modelName, versionId, minWorkers, maxWorkers, false, false);
    }

    public Map<String, Model> getDefaultModels(boolean skipFuntions) {
        ConcurrentHashMap<String, Model> defModelsMap = new ConcurrentHashMap<>();
        for (String key : modelsNameMap.keySet()) {
            ModelVersionedRefs mvr = modelsNameMap.get(key);
            if (mvr != null) {
                Model defaultModel = mvr.getDefaultModel();
                if (defaultModel != null) {
                    if (skipFuntions && defaultModel.getModelUrl() == null) {
                        continue;
                    }
                    defModelsMap.put(key, defaultModel);
                }
            }
        }
        return defModelsMap;
    }

    public Map<String, Model> getDefaultModels() {
        return getDefaultModels(false);
    }

    public List<WorkerThread> getWorkers(ModelVersionName modelVersionName) {
        return wlm.getWorkers(modelVersionName);
    }

    public Map<Integer, WorkerThread> getWorkers() {
        return wlm.getWorkers();
    }

    public boolean addJob(Job job) throws ModelNotFoundException, ModelVersionNotFoundException {
        String modelName = job.getModelName();
        String versionId = job.getModelVersion();
        Model model = getModel(modelName, versionId);
        if (model == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        if (wlm.hasNoWorker(model.getModelVersionName())) {
            return false;
        }

        return model.addJob(job);
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

    public Model getModel(String modelName, String versionId) throws ModelVersionNotFoundException {
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            return null;
        }
        Model model = vmodel.getVersionModel(versionId);
        if (model == null) {
            throw new ModelVersionNotFoundException(
                    "Model version: " + versionId + " does not exist for model: " + modelName);
        } else {
            return model;
        }
    }

    public Set<Entry<String, Model>> getAllModelVersions(String modelName)
            throws ModelNotFoundException {
        ModelVersionedRefs vmodel = modelsNameMap.get(modelName);
        if (vmodel == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        return vmodel.getAllVersions();
    }

    public Set<Entry<String, ModelVersionedRefs>> getAllModels() {
        return modelsNameMap.entrySet();
    }

    public int getNumRunningWorkers(ModelVersionName modelVersionName) {
        return wlm.getNumRunningWorkers(modelVersionName);
    }
}
