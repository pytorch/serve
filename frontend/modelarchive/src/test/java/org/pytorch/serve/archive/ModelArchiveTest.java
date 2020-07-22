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
    private static final List<String> VALID_HOSTS_LIST = Collections.singletonList("http(s)?://.*");

    @BeforeTest
    public void beforeTest() {
        output = new File("build/tmp/test/noop.mar");
        FileUtils.deleteQuietly(output);
        FileUtils.deleteQuietly(new File("build/tmp/test/noop"));
        FileUtils.deleteQuietly(new File("build/tmp/test/noop-v1.0.mar"));
        File tmp = FileUtils.getTempDirectory();
        FileUtils.deleteQuietly(new File(tmp, "models"));
    }

    @Test
    public void test() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive archive = ModelArchive.downloadModel(VALID_HOSTS_LIST, modelStore, "noop.mar");
        archive.validate();
        archive.clean();
        Assert.assertEquals(archive.getModelName(), "noop");
    }

    @Test(expectedExceptions = DownloadModelException.class)
    public void testInvalidURL() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(
                VALID_HOSTS_LIST,
                modelStore,
                "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.mod");
    }

    @Test(expectedExceptions = DownloadModelException.class)
    public void testMalformURL() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(
                VALID_HOSTS_LIST,
                modelStore,
                "https://../model-server/models/squeezenet_v1.1/squeezenet_v1.1.mod");
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void testValidHostURL() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        final List<String> customUrlPatternList =
                Collections.singletonList("http(s)?://s3.amazonaws.com.*");
        ModelArchive.downloadModel(
                customUrlPatternList,
                modelStore,
                "https://torchserve.s3.amazonaws.com/mar_files/mnist.mar");
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void testRelativePath() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(VALID_HOSTS_LIST, modelStore, "../mnist.mar");
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void testNullModelstore() throws ModelException, IOException {
        String modelStore = null;
        ModelArchive.downloadModel(VALID_HOSTS_LIST, modelStore, "../mnist.mar");
    }

    @Test(expectedExceptions = ModelNotFoundException.class)
    public void testMarFileNotexist() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive archive =
                ModelArchive.downloadModel(VALID_HOSTS_LIST, modelStore, "noop1.mar");
    }

    @Test(expectedExceptions = FileAlreadyExistsException.class)
    public void testFileAlreadyExist() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive archive =
                ModelArchive.downloadModel(
                        VALID_HOSTS_LIST,
                        modelStore,
                        "https://torchserve.s3.amazonaws.com/mar_files/mnist.mar");
    }
}
