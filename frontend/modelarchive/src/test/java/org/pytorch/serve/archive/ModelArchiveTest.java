package org.pytorch.serve.archive;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.util.Collections;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class ModelArchiveTest {

    private File output;
    private static final List<String> URL_PATTERN_LIST =
            Collections.singletonList("https://s3.amazonaws.com.*");

    @BeforeTest
    public void beforeTest() {
        output = new File("build/tmp/test/noop.mar");
        FileUtils.deleteQuietly(output);
        FileUtils.deleteQuietly(new File("build/tmp/test/noop"));
        FileUtils.deleteQuietly(new File("build/tmp/test/noop-v1.0.mar"));
        File tmp = FileUtils.getTempDirectory();
        FileUtils.deleteQuietly(new File(tmp, "models"));
    }

    @Test(expectedExceptions = FileAlreadyExistsException.class)
    public void test() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";

        ModelArchive archive = ModelArchive.downloadModel(URL_PATTERN_LIST, modelStore, "noop.mar");
        archive.validate();
        archive.clean();
        Assert.assertEquals(archive.getModelName(), "noop");

        // load model for s3 --> This will fail as this model is not compatible with
        // new implementation.
        // TODO Change this once we have example models on s3
        archive =
                ModelArchive.downloadModel(
                        URL_PATTERN_LIST,
                        modelStore,
                        "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model");
        Assert.assertEquals(archive.getModelName(), null);
        ModelArchive.downloadModel(
                URL_PATTERN_LIST,
                modelStore,
                "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model");
        ModelArchive.removeModel(
                modelStore,
                "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model");
        Assert.assertTrue(!new File(modelStore, "squeezenet_v1.1.model").exists());
    }

    @Test(expectedExceptions = DownloadModelException.class)
    public void testInvalidURL() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        // load model for s3 --> This will fail as this model is not compatible with
        // new implementation.
        // TODO Change this once we have example models on s3
        ModelArchive.downloadModel(
                URL_PATTERN_LIST,
                modelStore,
                "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.mod");
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void testMalformURL() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        // load model for s3 --> This will fail as this model is not compatible with
        // new implementation.
        // TODO Change this once we have example models on s3
        ModelArchive.downloadModel(
                URL_PATTERN_LIST,
                modelStore,
                "https://../model-server/models/squeezenet_v1.1/squeezenet_v1.1.mod");
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void testWhitelistURL() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(
                URL_PATTERN_LIST,
                modelStore,
                "https://torchserve.s3.amazonaws.com/mar_files/mnist.mar");
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void testRelativePath() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(URL_PATTERN_LIST, modelStore, "../mnist.mar");
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void testNullModelstore() throws ModelException, IOException {
        String modelStore = null;
        ModelArchive.downloadModel(URL_PATTERN_LIST, modelStore, "../mnist.mar");
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void testMarFileNotexist() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive archive =
                ModelArchive.downloadModel(URL_PATTERN_LIST, modelStore, "noop1.mar");
    }
}
