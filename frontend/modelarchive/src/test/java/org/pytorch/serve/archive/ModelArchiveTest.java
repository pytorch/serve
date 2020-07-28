package org.pytorch.serve.archive;

import java.io.File;
import java.io.IOException;
import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class ModelArchiveTest {

    private File output;

    @BeforeTest
    public void beforeTest() {
        output = new File("build/tmp/test/noop.mar");
        FileUtils.deleteQuietly(output);
        FileUtils.deleteQuietly(new File("build/tmp/test/noop"));
        FileUtils.deleteQuietly(new File("build/tmp/test/noop-v1.0.mar"));
        File tmp = FileUtils.getTempDirectory();
        FileUtils.deleteQuietly(new File(tmp, "models"));
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void test() throws ModelException, IOException, InterruptedException {
        String modelStore = "src/test/resources/models";

        ModelArchive archive = ModelArchive.downloadModel(modelStore, "noop.mar");
        archive.validate();
        archive.clean();
        Assert.assertEquals(archive.getModelName(), "noop");
        archive =
                ModelArchive.downloadModel(
                        modelStore,
                        "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model");
        Assert.assertEquals(archive.getModelName(), null);
        ModelArchive.removeModel(
                modelStore,
                "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model");
        Assert.assertTrue(!new File(modelStore, "squeezenet_v1.1.model").exists());
        ModelArchive.downloadModel(modelStore, "/../noop-v1.0");
    }

    @Test
    public void archiveTest() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive archive = ModelArchive.downloadModel(modelStore, "noop.mar");

        archive.getManifest().getModel().setModelVersion(null);
        Assert.assertThrows(InvalidModelException.class, () -> archive.validate());

        archive.getManifest().getModel().setModelName(null);
        Assert.assertThrows(InvalidModelException.class, () -> archive.validate());

        archive.getManifest().setModel(null);
        Assert.assertThrows(InvalidModelException.class, () -> archive.validate());

        archive.getManifest().setRuntime(null);
        Assert.assertThrows(InvalidModelException.class, () -> archive.validate());

        archive.getManifest().setRuntime(null);
        Assert.assertThrows(InvalidModelException.class, () -> archive.validate());

        Assert.assertThrows(
                ModelNotFoundException.class, () -> archive.downloadModel(null, "/noop"));

        Assert.assertThrows(
                ModelNotFoundException.class, () -> archive.downloadModel(modelStore, "../noop"));

        Assert.assertThrows(
                ModelNotFoundException.class, () -> archive.downloadModel("null", "/noop"));

        Assert.assertThrows(
                ModelNotFoundException.class,
                () -> ModelArchive.downloadModel("src/test/resources/", "models"));

        archive.clean();
    }

    @Test(expectedExceptions = DownloadModelException.class)
    public void testInvalidURL() throws ModelException, IOException, InterruptedException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(
                modelStore,
                "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.mod");
    }

    @Test(expectedExceptions = DownloadModelException.class)
    public void testMalformURL() throws ModelException, IOException, InterruptedException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(
                modelStore, "https://../model-server/models/squeezenet_v1.1/squeezenet_v1.1.mod");
    }
}
