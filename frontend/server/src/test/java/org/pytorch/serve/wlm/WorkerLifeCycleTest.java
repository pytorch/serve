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
        ConfigManager.Arguments args = new ConfigManager.Arguments();
        args.setModelStore("../archive/src/test/resources/models");
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
        Assert.assertEquals(archiveMnist.getModelName(), "mnist_scripted");
        Assert.assertEquals(archiveMnist.getModelVersion(), "1.0");
        WorkerLifeCycle workerLifeCycleMnist = new WorkerLifeCycle(configManager, modelMnist);
        workerLifeCycleMnist.startWorker(configManager.getInitialWorkerPort(), "");

        Assert.assertTrue(workerLifeCycleMnist.getProcess().isAlive());

        workerLifeCycleMnist.exit();
    }

    @Test
    public void testStartWorkerCppMnist()
            throws ModelException, IOException, DownloadArchiveException,
                    WorkerInitializationException, InterruptedException {
        ModelArchive archiveMnist =
                ModelArchive.downloadModel(
                        ALLOWED_URLS_LIST, configManager.getModelStore(), "mnist_scripted_cpp.mar");
        Model modelMnist = new Model(archiveMnist, 100);
        Assert.assertEquals(archiveMnist.getModelName(), "mnist_scripted_v2");
        Assert.assertEquals(archiveMnist.getModelVersion(), "2.0");

        modelMnist.setRuntimeType(
                configManager.getJsonRuntimeTypeValue(
                        archiveMnist.getModelName(),
                        archiveMnist.getModelVersion(),
                        Model.RUNTIME_TYPE,
                        archiveMnist.getManifest().getRuntime()));
        Assert.assertEquals(modelMnist.getRuntimeType().getValue(), LSP.getValue());

        WorkerLifeCycle workerLifeCycleMnist = new WorkerLifeCycle(configManager, modelMnist);
        workerLifeCycleMnist.startWorker(configManager.getInitialWorkerPort(), "");

        Assert.assertTrue(workerLifeCycleMnist.getProcess().isAlive());

        workerLifeCycleMnist.exit();
    }
}
