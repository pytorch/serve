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
    public void test() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";

        ModelArchive archive = ModelArchive.downloadModel(modelStore, "noop.mar");
        archive.validate();
        archive.clean();
        Assert.assertEquals(archive.getModelName(), "noop");

        // load model for s3 --> This will fail as this model is not compatible with
        // new implementation.
        // TODO Change this once we have example models on s3
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

    @Test(expectedExceptions = DownloadModelException.class)
    public void testInvalidURL() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        // load model for s3 --> This will fail as this model is not compatible with
        // new implementation.
        // TODO Change this once we have example models on s3
        ModelArchive.downloadModel(
                modelStore,
                "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.mod");
    }

    @Test(expectedExceptions = DownloadModelException.class)
    public void testMalformURL() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        // load model for s3 --> This will fail as this model is not compatible with
        // new implementation.
        // TODO Change this once we have example models on s3
        ModelArchive.downloadModel(
                modelStore, "https://../model-server/models/squeezenet_v1.1/squeezenet_v1.1.mod");
    }
}
