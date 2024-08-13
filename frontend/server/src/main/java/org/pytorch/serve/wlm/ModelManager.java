package org.pytorch.serve.wlm;

import com.google.gson.JsonObject;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
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
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.Manifest;
import org.pytorch.serve.archive.model.ModelArchive;
import org.pytorch.serve.archive.model.ModelConfig;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.http.ConflictStatusException;
import org.pytorch.serve.http.InvalidModelVersionException;
import org.pytorch.serve.http.messages.RegisterModelRequest;
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

    public static void init(ConfigManager configManager, WorkLoadManager wlm) {
        modelManager = new ModelManager(configManager, wlm);
    }

    public static ModelManager getInstance() {
        return modelManager;
    }

    public ScheduledExecutorService getScheduler() {
        return scheduler;
    }

    public ModelArchive registerModel(String url, String defaultModelName)
            throws ModelException, IOException, InterruptedException, DownloadArchiveException {
        return registerModel(
                url,
                null,
                null,
                null,
                -1 * RegisterModelRequest.DEFAULT_BATCH_SIZE,
                -1 * RegisterModelRequest.DEFAULT_MAX_BATCH_DELAY,
                configManager.getDefaultResponseTimeout(),
                configManager.getDefaultStartupTimeout(),
                defaultModelName,
                false,
                false,
                false);
    }

    public void registerAndUpdateModel(String modelName, JsonObject modelInfo)
            throws ModelException, IOException, InterruptedException, DownloadArchiveException,
                    WorkerInitializationException {

        boolean defaultVersion = modelInfo.get(Model.DEFAULT_VERSION).getAsBoolean();
        String url = modelInfo.get(Model.MAR_NAME).getAsString();

        ModelArchive archive = createModelArchive(modelName, url, null, null, modelName, false);

        Model tempModel = createModel(archive, modelInfo);

        String versionId = archive.getModelVersion();

        createVersionedModel(tempModel, versionId);

        setupModelVenv(tempModel);

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
            int startupTimeout,
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
            manifest.getModel().setEnvelope(configManager.getTsServiceEnvelope());
            File f = new File(handler.substring(0, handler.lastIndexOf(':')));
            archive = new ModelArchive(manifest, url, f.getParentFile(), true);
        } else {
            archive =
                    createModelArchive(
                            modelName, url, handler, runtime, defaultModelName, s3SseKms);
        }

        Model tempModel =
                createModel(
                        archive,
                        batchSize,
                        maxBatchDelay,
                        responseTimeout,
                        startupTimeout,
                        isWorkflowModel);

        String versionId = archive.getModelVersion();

        try {
            createVersionedModel(tempModel, versionId);
        } catch (ConflictStatusException e) {
            if (!ignoreDuplicate) {
                throw e;
            }
        }

        setupModelVenv(tempModel);

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
        Manifest.Model model = archive.getManifest().getModel();
        if (modelName == null || modelName.isEmpty()) {
            if (archive.getModelName() == null || archive.getModelName().isEmpty()) {
                model.setModelName(defaultModelName);
            }
        } else {
            model.setModelName(modelName);
        }

        if (runtime != null) {
            archive.getManifest().setRuntime(runtime);
        }

        if (handler != null) {
            model.setHandler(handler);
        } else if (archive.getHandler() == null || archive.getHandler().isEmpty()) {
            model.setHandler(configManager.getTsDefaultServiceHandler());
        }

        model.setEnvelope(configManager.getTsServiceEnvelope());

        if (model.getModelVersion() == null) {
            model.setModelVersion("1.0");
        }

        archive.validate();

        return archive;
    }

    private void setupModelVenv(Model model)
            throws IOException, InterruptedException, ModelException {
        if (!model.isUseVenv()) {
            return;
        }

        File venvPath = EnvironmentUtils.getPythonVenvPath(model);
        List<String> commandParts = new ArrayList<>();
        commandParts.add(configManager.getPythonExecutable());
        commandParts.add(
                Paths.get(configManager.getModelServerHome(), "ts", "utils", "setup_model_venv.py")
                        .toAbsolutePath()
                        .toString());
        commandParts.add(venvPath.toString());

        ProcessBuilder processBuilder = new ProcessBuilder(commandParts);

        if (isValidDependencyPath(venvPath)) {
            processBuilder.directory(venvPath.getParentFile());
        } else {
            throw new ModelException(
                    "Invalid python venv path for model "
                            + model.getModelName()
                            + ": "
                            + venvPath.toString());
        }
        Map<String, String> environment = processBuilder.environment();
        String[] envp =
                EnvironmentUtils.getEnvString(
                        configManager.getModelServerHome(),
                        model.getModelDir().getAbsolutePath(),
                        null);
        for (String envVar : envp) {
            String[] parts = envVar.split("=", 2);
            if (parts.length == 2) {
                environment.put(parts[0], parts[1]);
            }
        }
        processBuilder.redirectErrorStream(true);

        Process process = processBuilder.start();

        int exitCode = process.waitFor();
        String line;
        StringBuilder outputString = new StringBuilder();
        BufferedReader brdr = new BufferedReader(new InputStreamReader(process.getInputStream()));
        while ((line = brdr.readLine()) != null) {
            outputString.append(line + "\n");
        }

        if (exitCode == 0) {
            logger.info(
                    "Created virtual environment for model {}: {}",
                    model.getModelName(),
                    venvPath.toString());
        } else {
            logger.error(
                    "Virtual environment creation for model {} at {} failed:\n{}",
                    model.getModelName(),
                    venvPath.toString(),
                    outputString.toString());
            throw new ModelException(
                    "Virtual environment creation failed for model " + model.getModelName());
        }
    }

    private void setupModelDependencies(Model model)
            throws IOException, InterruptedException, ModelException {
        String requirementsFile =
                model.getModelArchive().getManifest().getModel().getRequirementsFile();

        if (!configManager.getInstallPyDepPerModel() || requirementsFile == null) {
            return;
        }

        String pythonRuntime = EnvironmentUtils.getPythonRunTime(model);
        Path requirementsFilePath =
                Paths.get(model.getModelDir().getAbsolutePath(), requirementsFile).toAbsolutePath();
        List<String> commandParts = new ArrayList<>();
        ProcessBuilder processBuilder = new ProcessBuilder();

        if (model.isUseVenv()) {
            if (!isValidDependencyPath(Paths.get(pythonRuntime).toFile())) {
                throw new ModelException(
                        "Invalid python venv runtime path for model "
                                + model.getModelName()
                                + ": "
                                + pythonRuntime);
            }

            processBuilder.directory(EnvironmentUtils.getPythonVenvPath(model).getParentFile());

            commandParts.add(pythonRuntime);
            commandParts.add("-m");
            commandParts.add("pip");
            commandParts.add("install");
            commandParts.add("-U");
            commandParts.add("--upgrade-strategy");
            commandParts.add("only-if-needed");
            commandParts.add("-r");
            commandParts.add(requirementsFilePath.toString());
        } else {
            File dependencyPath = model.getModelDir();
            if (Files.isSymbolicLink(dependencyPath.toPath())) {
                dependencyPath = dependencyPath.getParentFile();
            }
            dependencyPath = dependencyPath.getAbsoluteFile();
            if (!isValidDependencyPath(dependencyPath)) {
                throw new ModelException(
                        "Invalid 3rd party package installation path " + dependencyPath.toString());
            }

            processBuilder.directory(dependencyPath);

            commandParts.add(pythonRuntime);
            commandParts.add("-m");
            commandParts.add("pip");
            commandParts.add("install");
            commandParts.add("-U");
            commandParts.add("-t");
            commandParts.add(dependencyPath.toString());
            commandParts.add("-r");
            commandParts.add(requirementsFilePath.toString());
        }

        processBuilder.command(commandParts);
        String[] envp =
                EnvironmentUtils.getEnvString(
                        configManager.getModelServerHome(),
                        model.getModelDir().getAbsolutePath(),
                        null);
        Map<String, String> environment = processBuilder.environment();
        for (String envVar : envp) {
            String[] parts = envVar.split("=", 2);
            if (parts.length == 2) {
                environment.put(parts[0], parts[1]);
            }
        }
        processBuilder.redirectErrorStream(true);

        Process process = processBuilder.start();

        int exitCode = process.waitFor();
        String line;
        StringBuilder outputString = new StringBuilder();
        BufferedReader brdr = new BufferedReader(new InputStreamReader(process.getInputStream()));
        while ((line = brdr.readLine()) != null) {
            outputString.append(line + "\n");
        }

        if (exitCode == 0) {
            logger.info("Installed custom pip packages for model {}", model.getModelName());
        } else {
            logger.error(
                    "Custom pip package installation failed for model {}:\n{}",
                    model.getModelName(),
                    outputString.toString());
            throw new ModelException(
                    "Custom pip package installation failed for model " + model.getModelName());
        }
    }

    private boolean isValidDependencyPath(File dependencyPath) {
        if (dependencyPath
                .toPath()
                .normalize()
                .startsWith(FileUtils.getTempDirectory().toPath().normalize())) {
            return true;
        }
        return false;
    }

    private Model createModel(
            ModelArchive archive,
            int batchSize,
            int maxBatchDelay,
            int responseTimeout,
            int startupTimeout,
            boolean isWorkflowModel) {
        Model model = new Model(archive, configManager.getJobQueueSize());

        if (batchSize == -1 * RegisterModelRequest.DEFAULT_BATCH_SIZE) {
            if (archive.getModelConfig() != null) {
                int marBatchSize = archive.getModelConfig().getBatchSize();
                batchSize =
                        marBatchSize > 0
                                ? marBatchSize
                                : configManager.getJsonIntValue(
                                        archive.getModelName(),
                                        archive.getModelVersion(),
                                        Model.BATCH_SIZE,
                                        RegisterModelRequest.DEFAULT_BATCH_SIZE);
            } else {
                batchSize =
                        configManager.getJsonIntValue(
                                archive.getModelName(),
                                archive.getModelVersion(),
                                Model.BATCH_SIZE,
                                RegisterModelRequest.DEFAULT_BATCH_SIZE);
            }
        }
        model.setBatchSize(batchSize);

        if (maxBatchDelay == -1 * RegisterModelRequest.DEFAULT_MAX_BATCH_DELAY) {
            if (archive.getModelConfig() != null) {
                int marMaxBatchDelay = archive.getModelConfig().getMaxBatchDelay();
                maxBatchDelay =
                        marMaxBatchDelay > 0
                                ? marMaxBatchDelay
                                : configManager.getJsonIntValue(
                                        archive.getModelName(),
                                        archive.getModelVersion(),
                                        Model.MAX_BATCH_DELAY,
                                        RegisterModelRequest.DEFAULT_MAX_BATCH_DELAY);
            } else {
                maxBatchDelay =
                        configManager.getJsonIntValue(
                                archive.getModelName(),
                                archive.getModelVersion(),
                                Model.MAX_BATCH_DELAY,
                                RegisterModelRequest.DEFAULT_MAX_BATCH_DELAY);
            }
        }
        model.setMaxBatchDelay(maxBatchDelay);

        if (archive.getModelConfig() != null) {
            int marResponseTimeout = archive.getModelConfig().getResponseTimeout();
            int marStartupTimeout = archive.getModelConfig().getStartupTimeout();
            responseTimeout =
                    marResponseTimeout > 0
                            ? marResponseTimeout
                            : configManager.getJsonIntValue(
                                    archive.getModelName(),
                                    archive.getModelVersion(),
                                    Model.RESPONSE_TIMEOUT,
                                    responseTimeout);
            startupTimeout =
                    marStartupTimeout > 0
                            ? marStartupTimeout
                            : configManager.getJsonIntValue(
                                    archive.getModelName(),
                                    archive.getModelVersion(),
                                    Model.STARTUP_TIMEOUT,
                                    startupTimeout);
        } else {
            responseTimeout =
                    configManager.getJsonIntValue(
                            archive.getModelName(),
                            archive.getModelVersion(),
                            Model.RESPONSE_TIMEOUT,
                            responseTimeout);
            startupTimeout =
                    configManager.getJsonIntValue(
                            archive.getModelName(),
                            archive.getModelVersion(),
                            Model.STARTUP_TIMEOUT,
                            startupTimeout);
        }
        model.setResponseTimeout(responseTimeout);
        model.setStartupTimeout(startupTimeout);
        model.setWorkflowModel(isWorkflowModel);
        model.setRuntimeType(
                configManager.getJsonRuntimeTypeValue(
                        archive.getModelName(),
                        archive.getModelVersion(),
                        Model.RUNTIME_TYPE,
                        archive.getManifest().getRuntime()));

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
            throws ModelVersionNotFoundException, WorkerInitializationException {
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
            throws ModelVersionNotFoundException, WorkerInitializationException {
        Model model = getVersionModel(modelName, versionId);

        if (model == null) {
            throw new ModelVersionNotFoundException(
                    "Model version: " + versionId + " does not exist for model: " + modelName);
        }
        if (model.getParallelLevel() > 0 && model.getDeviceType() == ModelConfig.DeviceType.GPU) {
            /**
             * Current capacity check for LMI is based on single node. TODO: multiple nodes check
             * will be based on --proc-per-node + numCores.
             */
            int capacity = model.getNumCores() / model.getParallelLevel();
            if (capacity == 0) {
                logger.error(
                        "there are no enough gpu devices to support this parallelLever: {}",
                        model.getParallelLevel());
                throw new WorkerInitializationException(
                        "No enough gpu devices for model:"
                                + modelName
                                + " parallelLevel:"
                                + model.getParallelLevel());
            } else {
                minWorkers = minWorkers > capacity ? capacity : minWorkers;
                maxWorkers = maxWorkers > capacity ? capacity : maxWorkers;
                logger.info(
                        "model {} set minWorkers: {}, maxWorkers: {} for parallelLevel: {} ",
                        modelName,
                        minWorkers,
                        maxWorkers,
                        model.getParallelLevel());
            }
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
            throws ModelVersionNotFoundException, WorkerInitializationException {
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

    public int getNumHealthyWorkers(ModelVersionName modelVersionName) {
        return wlm.getNumHealthyWorkers(modelVersionName);
    }
}
