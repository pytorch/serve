package org.pytorch.serve.util;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.HttpResponseStatus;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.nio.file.FileAlreadyExistsException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.function.Function;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.Manifest;
import org.pytorch.serve.archive.model.ModelArchive;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.archive.utils.ArchiveUtils;
import org.pytorch.serve.http.BadRequestException;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.http.InvalidModelVersionException;
import org.pytorch.serve.http.RequestTimeoutException;
import org.pytorch.serve.http.ServiceUnavailableException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.http.messages.DescribeModelResponse;
import org.pytorch.serve.http.messages.ListModelsResponse;
import org.pytorch.serve.http.messages.RegisterModelRequest;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.snapshot.SnapshotManager;
import org.pytorch.serve.util.messages.RequestInput;
import org.pytorch.serve.util.messages.WorkerCommands;
import org.pytorch.serve.wlm.Model;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.ModelVersionedRefs;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.pytorch.serve.wlm.WorkerState;
import org.pytorch.serve.wlm.WorkerThread;

public final class ApiUtils {

    private ApiUtils() {}

    public static ListModelsResponse getModelList(int limit, int pageToken) {
        if (limit > 100 || limit < 0) {
            limit = 100;
        }
        if (pageToken < 0) {
            pageToken = 0;
        }

        Map<String, Model> models = ModelManager.getInstance().getDefaultModels(true);
        List<String> keys = new ArrayList<>(models.keySet());
        Collections.sort(keys);
        ListModelsResponse list = new ListModelsResponse();

        int last = pageToken + limit;
        if (last > keys.size()) {
            last = keys.size();
        } else {
            list.setNextPageToken(String.valueOf(last));
        }

        for (int i = pageToken; i < last; ++i) {
            String modelName = keys.get(i);
            Model model = models.get(modelName);
            list.addModel(modelName, model.getModelUrl());
        }

        return list;
    }

    public static ArrayList<DescribeModelResponse> getModelDescription(
            String modelName, String modelVersion)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        ArrayList<DescribeModelResponse> resp = new ArrayList<DescribeModelResponse>();

        if ("all".equals(modelVersion)) {
            for (Map.Entry<String, Model> m : modelManager.getAllModelVersions(modelName)) {
                resp.add(createModelResponse(modelManager, modelName, m.getValue()));
            }
        } else {
            Model model = modelManager.getModel(modelName, modelVersion);
            if (model == null) {
                throw new ModelNotFoundException("Model not found: " + modelName);
            }
            resp.add(createModelResponse(modelManager, modelName, model));
        }

        return resp;
    }

    public static String setDefault(String modelName, String newModelVersion)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        modelManager.setDefaultVersion(modelName, newModelVersion);
        String msg =
                "Default version successfully updated for model \""
                        + modelName
                        + "\" to \""
                        + newModelVersion
                        + "\"";
        SnapshotManager.getInstance().saveSnapshot();
        return msg;
    }

    public static StatusResponse registerModel(RegisterModelRequest registerModelRequest)
            throws ModelException, InternalServerException, ExecutionException,
                    InterruptedException, DownloadArchiveException, WorkerInitializationException {
        String modelUrl = registerModelRequest.getModelUrl();
        if (modelUrl == null) {
            throw new BadRequestException("Parameter url is required.");
        }

        String modelName = registerModelRequest.getModelName();
        String runtime = registerModelRequest.getRuntime();
        String handler = registerModelRequest.getHandler();
        int batchSize = registerModelRequest.getBatchSize();
        int maxBatchDelay = registerModelRequest.getMaxBatchDelay();
        int initialWorkers = registerModelRequest.getInitialWorkers();
        int responseTimeout = registerModelRequest.getResponseTimeout();
        int startupTimeout = registerModelRequest.getStartupTimeout();
        boolean s3SseKms = registerModelRequest.getS3SseKms();
        if (responseTimeout == -1) {
            responseTimeout = ConfigManager.getInstance().getDefaultResponseTimeout();
        }
        if (startupTimeout == -1) {
            startupTimeout = ConfigManager.getInstance().getDefaultStartupTimeout();
        }

        Manifest.RuntimeType runtimeType = null;
        if (runtime != null) {
            try {
                runtimeType = Manifest.RuntimeType.fromValue(runtime);
            } catch (IllegalArgumentException e) {
                throw new BadRequestException(e);
            }
        }

        return handleRegister(
                modelUrl,
                modelName,
                runtimeType,
                handler,
                batchSize,
                maxBatchDelay,
                responseTimeout,
                startupTimeout,
                initialWorkers,
                registerModelRequest.getSynchronous(),
                false,
                s3SseKms);
    }

    public static StatusResponse handleRegister(
            String modelUrl,
            String modelName,
            Manifest.RuntimeType runtimeType,
            String handler,
            int batchSize,
            int maxBatchDelay,
            int responseTimeout,
            int startupTimeout,
            int initialWorkers,
            boolean isSync,
            boolean isWorkflowModel,
            boolean s3SseKms)
            throws ModelException, ExecutionException, InterruptedException,
                    DownloadArchiveException, WorkerInitializationException {

        ModelManager modelManager = ModelManager.getInstance();
        final ModelArchive archive;
        try {
            archive =
                    modelManager.registerModel(
                            modelUrl,
                            modelName,
                            runtimeType,
                            handler,
                            batchSize,
                            maxBatchDelay,
                            responseTimeout,
                            startupTimeout,
                            null,
                            false,
                            isWorkflowModel,
                            s3SseKms);
        } catch (FileAlreadyExistsException e) {
            throw new InternalServerException(
                    "Model file already exists " + ArchiveUtils.getFilenameFromUrl(modelUrl), e);
        } catch (IOException | InterruptedException e) {
            throw new InternalServerException("Failed to save model: " + modelUrl, e);
        }

        modelName = archive.getModelName();
        int minWorkers = 0;
        int maxWorkers = 0;
        if (archive.getModelConfig() != null) {
            int marMinWorkers = archive.getModelConfig().getMinWorkers();
            int marMaxWorkers = archive.getModelConfig().getMaxWorkers();
            if (marMinWorkers > 0 && marMaxWorkers >= marMinWorkers) {
                minWorkers = marMinWorkers;
                maxWorkers = marMaxWorkers;
            }
        }
        if (initialWorkers <= 0 && minWorkers == 0) {
            final String msg =
                    "Model \""
                            + modelName
                            + "\" Version: "
                            + archive.getModelVersion()
                            + " registered with 0 initial workers. Use scale workers API to add workers for the model.";
            if (!isWorkflowModel) {
                SnapshotManager.getInstance().saveSnapshot();
            }
            return new StatusResponse(msg, HttpURLConnection.HTTP_OK);
        }
        minWorkers = minWorkers > 0 ? minWorkers : initialWorkers;
        maxWorkers = maxWorkers > 0 ? maxWorkers : initialWorkers;

        return ApiUtils.updateModelWorkers(
                modelName,
                archive.getModelVersion(),
                minWorkers,
                maxWorkers,
                isSync,
                true,
                f -> {
                    modelManager.unregisterModel(archive.getModelName(), archive.getModelVersion());
                    return null;
                });
    }

    public static StatusResponse updateModelWorkers(
            String modelName,
            String modelVersion,
            int minWorkers,
            int maxWorkers,
            boolean synchronous,
            boolean isInit,
            final Function<Void, Void> onError)
            throws ModelVersionNotFoundException, ModelNotFoundException, ExecutionException,
                    InterruptedException, WorkerInitializationException {

        ModelManager modelManager = ModelManager.getInstance();
        if (maxWorkers < minWorkers) {
            throw new BadRequestException("max_worker cannot be less than min_worker.");
        }
        if (!modelManager.getDefaultModels().containsKey(modelName)) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }

        CompletableFuture<Integer> future =
                modelManager.updateModel(modelName, modelVersion, minWorkers, maxWorkers);

        StatusResponse statusResponse = new StatusResponse();

        if (!synchronous) {
            return new StatusResponse(
                    "Processing worker updates...", HttpURLConnection.HTTP_ACCEPTED);
        }

        CompletableFuture<StatusResponse> statusResponseCompletableFuture =
                future.thenApply(
                                v -> {
                                    boolean status =
                                            modelManager.scaleRequestStatus(
                                                    modelName, modelVersion);

                                    if (HttpURLConnection.HTTP_OK == v) {
                                        if (status) {
                                            String msg =
                                                    "Workers scaled to "
                                                            + minWorkers
                                                            + " for model: "
                                                            + modelName;
                                            if (modelVersion != null) {
                                                msg += ", version: " + modelVersion; // NOPMD
                                            }

                                            if (isInit) {
                                                msg =
                                                        "Model \""
                                                                + modelName
                                                                + "\" Version: "
                                                                + modelVersion
                                                                + " registered with "
                                                                + minWorkers
                                                                + " initial workers";
                                            }

                                            statusResponse.setStatus(msg);
                                            statusResponse.setHttpResponseCode(v);
                                        } else {
                                            statusResponse.setStatus(
                                                    "Workers scaling in progress...");
                                            statusResponse.setHttpResponseCode(
                                                    HttpURLConnection.HTTP_PARTIAL);
                                        }
                                    } else {
                                        statusResponse.setHttpResponseCode(v);
                                        String msg =
                                                "Failed to start workers for model "
                                                        + modelName
                                                        + " version: "
                                                        + modelVersion;
                                        statusResponse.setStatus(msg);
                                        statusResponse.setE(new InternalServerException(msg));
                                        if (onError != null) {
                                            onError.apply(null);
                                        }
                                    }
                                    return statusResponse;
                                })
                        .exceptionally(
                                (e) -> {
                                    if (onError != null) {
                                        onError.apply(null);
                                    }
                                    statusResponse.setStatus(e.getMessage());
                                    statusResponse.setHttpResponseCode(
                                            HttpURLConnection.HTTP_INTERNAL_ERROR);
                                    statusResponse.setE(e);
                                    return statusResponse;
                                });

        return statusResponseCompletableFuture.get();
    }

    public static void unregisterModel(String modelName, String modelVersion)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        int httpResponseStatus = modelManager.unregisterModel(modelName, modelVersion);
        if (httpResponseStatus == HttpResponseStatus.NOT_FOUND.code()) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.BAD_REQUEST.code()) {
            throw new ModelVersionNotFoundException(
                    String.format(
                            "Model version: %s does not exist for model: %s",
                            modelVersion, modelName));
        } else if (httpResponseStatus == HttpResponseStatus.INTERNAL_SERVER_ERROR.code()) {
            throw new InternalServerException("Interrupted while cleaning resources: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.REQUEST_TIMEOUT.code()) {
            throw new RequestTimeoutException("Timed out while cleaning resources: " + modelName);
        } else if (httpResponseStatus == HttpResponseStatus.FORBIDDEN.code()) {
            throw new InvalidModelVersionException(
                    "Cannot remove default version for model " + modelName);
        }
    }

    public static void getTorchServeHealth(Runnable r) {
        ModelManager modelManager = ModelManager.getInstance();
        modelManager.submitTask(r);
    }

    public static String getWorkerStatus() {
        ModelManager modelManager = ModelManager.getInstance();
        String response = "Healthy";
        int numWorking = 0;
        int numScaled = 0;

        for (Map.Entry<String, ModelVersionedRefs> m : modelManager.getAllModels()) {
            numScaled += m.getValue().getDefaultModel().getMinWorkers();
            numWorking +=
                    modelManager.getNumRunningWorkers(
                            m.getValue().getDefaultModel().getModelVersionName());
        }

        if ((numWorking > 0) && (numWorking < numScaled)) {
            response = "Partial Healthy";
        } else if ((numWorking == 0) && (numScaled > 0)) {
            response = "Unhealthy";
        }
        // TODO: Check if its OK to send other 2xx errors to ALB for "Partial Healthy"
        // and
        // "Unhealthy"
        return response;
    }

    public static boolean isModelHealthy() {
        ModelManager modelManager = ModelManager.getInstance();
        int numHealthy = 0;
        int numScaled = 0;

        for (Map.Entry<String, ModelVersionedRefs> m : modelManager.getAllModels()) {
            numScaled = m.getValue().getDefaultModel().getMinWorkers();
            numHealthy =
                    modelManager.getNumHealthyWorkers(
                            m.getValue().getDefaultModel().getModelVersionName());
            if (numHealthy < numScaled) {
                return false;
            }
        }
        return true;
    }

    private static DescribeModelResponse createModelResponse(
            ModelManager modelManager, String modelName, Model model) {
        DescribeModelResponse resp = new DescribeModelResponse();
        resp.setModelName(modelName);
        resp.setModelUrl(model.getModelUrl());
        resp.setBatchSize(model.getBatchSize());
        resp.setMaxBatchDelay(model.getMaxBatchDelay());
        resp.setMaxWorkers(model.getMaxWorkers());
        resp.setMinWorkers(model.getMinWorkers());
        resp.setLoadedAtStartup(modelManager.getStartupModels().contains(modelName));
        Manifest manifest = model.getModelArchive().getManifest();
        resp.setModelVersion(manifest.getModel().getModelVersion());
        resp.setRuntime(manifest.getRuntime().getValue());
        resp.setResponseTimeout(model.getResponseTimeout());
        resp.setStartupTimeout(model.getStartupTimeout());
        resp.setMaxRetryTimeoutInSec(model.getMaxRetryTimeoutInMill() / 1000);
        resp.setClientTimeoutInMills(model.getClientTimeoutInMills());
        resp.setParallelType(model.getParallelType().getParallelType());
        resp.setParallelLevel(model.getParallelLevel());
        resp.setDeviceType(model.getDeviceType().getDeviceType());
        resp.setDeviceIds(model.getDeviceIds());
        resp.setContinuousBatching(model.isContinuousBatching());
        resp.setUseJobTicket(model.isUseJobTicket());
        resp.setUseVenv(model.isUseVenv());
        resp.setStateful(model.isSequenceBatching());
        resp.setSequenceMaxIdleMSec(model.getSequenceMaxIdleMSec());
        resp.setSequenceTimeoutMSec(model.getSequenceTimeoutMSec());
        resp.setMaxNumSequence(model.getMaxNumSequence());
        resp.setMaxSequenceJobQueueSize(model.getMaxSequenceJobQueueSize());

        List<WorkerThread> workers = modelManager.getWorkers(model.getModelVersionName());
        for (WorkerThread worker : workers) {
            String workerId = worker.getWorkerId();
            long startTime = worker.getStartTime();
            boolean isRunning =
                    worker.isRunning() && worker.getState() == WorkerState.WORKER_MODEL_LOADED;
            int gpuId = worker.getGpuId();
            long memory = worker.getMemory();
            int pid = worker.getPid();
            String gpuUsage = worker.getGpuUsage();
            resp.addWorker(workerId, startTime, isRunning, gpuId, memory, pid, gpuUsage);
        }

        DescribeModelResponse.JobQueueStatus jobQueueStatus =
                new DescribeModelResponse.JobQueueStatus();
        jobQueueStatus.setRemainingCapacity(model.getJobQueueRemainingCapacity());
        jobQueueStatus.setPendingRequests(model.getPendingRequestsInJobQueue());
        resp.setJobQueueStatus(jobQueueStatus);

        return resp;
    }

    public static RestJob addRESTInferenceJob(
            ChannelHandlerContext ctx, String modelName, String version, RequestInput input)
            throws ModelNotFoundException, ModelVersionNotFoundException {
        RestJob job = new RestJob(ctx, modelName, version, WorkerCommands.PREDICT, input);
        if (!ModelManager.getInstance().addJob(job)) {
            String responseMessage = getStreamingInferenceErrorResponseMessage(modelName, version);
            throw new ServiceUnavailableException(responseMessage);
        }
        return job;
    }

    @SuppressWarnings("PMD")
    public static String getStreamingInferenceErrorResponseMessage(
            String modelName, String modelVersion) {
        StringBuilder responseMessage = new StringBuilder().append("Model \"").append(modelName);

        if (modelVersion != null) {
            responseMessage.append("\" Version ").append(modelVersion);
        }

        responseMessage.append(
                "\" has no worker to serve inference request. Please use scale workers API to add workers. "
                        + "If this is a sequence inference, please check if it is closed, or expired;"
                        + " or exceeds maxSequenceJobQueueSize");
        return responseMessage.toString();
    }

    public static String getDescribeErrorResponseMessage(String modelName) {
        String responseMessage = "Model \"" + modelName;

        responseMessage +=
                "\" has no worker to serve describe request. Please use scale workers API to add workers.";
        return responseMessage;
    }
}
