package org.pytorch.serve.wlm;

import static org.pytorch.serve.archive.model.Manifest.RuntimeType.LSP;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.ModelArchive;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.util.ConfigManager;
import org.testng.Assert;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

public class WorkerLifeCycleTest {
    private static final List<String> ALLOWED_URLS_LIST =
            Collections.singletonList("file://.*|http(s)?://.*");
    private ConfigManager configManager;

    @BeforeSuite
    public void beforeSuite() throws IOException {
        System.setProperty("tsConfigFile", "src/test/resources/config_test_cpp.properties");
        ConfigManager.Arguments args = new ConfigManager.Arguments();
        // args.setModels(new String[] {"noop_v0.1"});
        args.setSnapshotDisabled(true);
        ConfigManager.init(args);
        configManager = ConfigManager.getInstance();
    }

    @Test
    public void testStartWorkerNoop() throws ModelException, IOException, DownloadArchiveException {
        ModelArchive archiveNoop =
                ModelArchive.downloadModel(
                        ALLOWED_URLS_LIST, configManager.getModelStore(), "noop.mar");
        Model modelNoop = new Model(archiveNoop, 100);
        Assert.assertEquals(modelNoop.getRuntimeType().getValue(), "python");
    }

    @Test
    public void testStartWorkerPythonMnist()
            throws ModelException, IOException, DownloadArchiveException,
                    WorkerInitializationException, InterruptedException {
        ModelArchive archiveMnist =
                ModelArchive.downloadModel(
                        ALLOWED_URLS_LIST, configManager.getModelStore(), "mnist_scripted.mar");
        Model modelMnist = new Model(archiveMnist, 100);
        Assert.assertEquals(archiveMnist.getModelName(), "mnist_ts");
        Assert.assertEquals(archiveMnist.getModelVersion(), "1.0");
        WorkerLifeCycle workerLifeCycleMnist = new WorkerLifeCycle(configManager, modelMnist);
        workerLifeCycleMnist.startWorker(configManager.getInitialWorkerPort());
    }

    @Test
    public void testStartWorkerCppMnist()
            throws ModelException, IOException, DownloadArchiveException,
                    WorkerInitializationException, InterruptedException {
        ModelArchive archiveMnist =
                ModelArchive.downloadModel(
                        ALLOWED_URLS_LIST, configManager.getModelStore(), "mnist_scripted.mar");
        Model modelMnist = new Model(archiveMnist, 100);
        Assert.assertEquals(archiveMnist.getModelName(), "mnist_ts");
        Assert.assertEquals(archiveMnist.getModelVersion(), "1.0");

        modelMnist.setRuntimeType(
                configManager.getJsonRuntimeTypeValue(
                        archiveMnist.getModelName(),
                        archiveMnist.getModelVersion(),
                        Model.RUNTIME_TYPE,
                        archiveMnist.getManifest().getRuntime()));
        Assert.assertEquals(modelMnist.getRuntimeType().getValue(), LSP.getValue());

        WorkerLifeCycle workerLifeCycleMnist = new WorkerLifeCycle(configManager, modelMnist);
        // workerLifeCycleMnist.startWorker(configManager.getInitialWorkerPort());
    }
}
