package org.pytorch.serve.archive;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class ModelArchiveTest {

    private static final List<String> ALLOWED_URLS_LIST =
            Collections.singletonList("file://.*|http(s)?://.*");

    @BeforeTest
    public void beforeTest() {
        File output = new File("build/tmp/test/noop.mar");
        FileUtils.deleteQuietly(output);
        FileUtils.deleteQuietly(new File("build/tmp/test/noop"));
        FileUtils.deleteQuietly(new File("build/tmp/test/noop-v1.0.mar"));
        File tmp = FileUtils.getTempDirectory();
        FileUtils.deleteQuietly(new File(tmp, "models"));
    }

    @Test
    public void test() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive archive =
                ModelArchive.downloadModel(ALLOWED_URLS_LIST, modelStore, "noop.mar");
        archive.validate();
        archive.clean();
        Assert.assertEquals(archive.getModelName(), "noop");
    }

    @Test(
            expectedExceptions = DownloadModelException.class,
            expectedExceptionsMessageRegExp =
                    "Failed to download model from: https://s3\\.amazonaws\\.com/squeezenet_v1\\.1\\.mod")
    public void testAllowedURL() throws ModelException, IOException {
        // test allowed url, return failed to download as file does not exist
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(
                ALLOWED_URLS_LIST, modelStore, "https://s3.amazonaws.com/squeezenet_v1.1.mod");
    }

    @Test(
            expectedExceptions = DownloadModelException.class,
            expectedExceptionsMessageRegExp =
                    "Failed to download model from: https://torchserve\\.pytorch\\.org/mar_files/mnist_non_exist\\.mar")
    public void testAllowedMultiUrls() throws ModelException, IOException {
        // test multiple urls added to allowed list
        String modelStore = "src/test/resources/models";
        final List<String> customUrlPatternList =
                Arrays.asList(
                        "http(s)?://s3.amazonaws.com.*",
                        "https://torchserve.pytorch.org/mar_files/.*");
        ModelArchive.downloadModel(
                customUrlPatternList,
                modelStore,
                "https://torchserve.pytorch.org/mar_files/mnist_non_exist.mar");
    }

    @Test(
            expectedExceptions = ModelNotFoundException.class,
            expectedExceptionsMessageRegExp =
                    "Given URL https://torchserve\\.pytorch.org/mar_files/mnist\\.mar does not match any allowed URL\\(s\\)")
    public void testBlockedUrl() throws ModelException, IOException {
        // test blocked url
        String modelStore = "src/test/resources/models";
        final List<String> customUrlPatternList =
                Collections.singletonList("http(s)?://s3.amazonaws.com.*");
        ModelArchive.downloadModel(
                customUrlPatternList,
                modelStore,
                "https://torchserve.pytorch.org/mar_files/mnist.mar");
    }

    @Test
    public void testLocalFile() throws ModelException, IOException, InterruptedException {
        String modelStore = "src/test/resources/models";
        String curDir = System.getProperty("user.dir");
        File curDirFile = new File(curDir);
        String parent = curDirFile.getParent();

        // Setup: This test needs mar file in local path. Copying mnist.mar from model folder.
        String source = modelStore + "/mnist.mar";
        String destination = parent + "/modelarchive/mnist1.mar";
        File sourceFile = new File(source);
        File destinationFile = new File(destination);
        FileUtils.copyFile(sourceFile, destinationFile);

        String fileUrl = "file:///" + parent + "/modelarchive/mnist1.mar";
        ModelArchive archive = ModelArchive.downloadModel(ALLOWED_URLS_LIST, modelStore, fileUrl);

        File modelLocation = new File(modelStore + "/mnist1.mar");
        Assert.assertTrue(modelLocation.exists());
        ModelArchive.removeModel(modelStore, fileUrl);
        Assert.assertTrue(!new File(modelStore, "mnist1").exists());
        FileUtils.deleteQuietly(destinationFile);
    }

    @Test
    public void archiveTest() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive archive =
                ModelArchive.downloadModel(ALLOWED_URLS_LIST, modelStore, "noop.mar");

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
                ModelNotFoundException.class,
                () -> archive.downloadModel(ALLOWED_URLS_LIST, null, "/noop"));

        Assert.assertThrows(
                ModelNotFoundException.class,
                () -> archive.downloadModel(ALLOWED_URLS_LIST, modelStore, "../noop"));

        Assert.assertThrows(
                ModelNotFoundException.class,
                () -> archive.downloadModel(ALLOWED_URLS_LIST, "null", "/noop"));

        Assert.assertThrows(
                ModelNotFoundException.class,
                () ->
                        ModelArchive.downloadModel(
                                ALLOWED_URLS_LIST, "src/test/resources/", "models"));

        archive.clean();
    }

    @Test(expectedExceptions = DownloadModelException.class)
    public void testMalformedURL() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(
                ALLOWED_URLS_LIST,
                modelStore,
                "https://../model-server/models/squeezenet_v1.1/squeezenet_v1.1.mod");
    }

    @Test(
            expectedExceptions = ModelNotFoundException.class,
            expectedExceptionsMessageRegExp =
                    "Relative path is not allowed in url: \\.\\./mnist\\.mar")
    public void testRelativePath() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(ALLOWED_URLS_LIST, modelStore, "../mnist.mar");
    }

    @Test(
            expectedExceptions = ModelNotFoundException.class,
            expectedExceptionsMessageRegExp = "Model store has not been configured\\.")
    public void testNullModelstore() throws ModelException, IOException {
        String modelStore = null;
        ModelArchive.downloadModel(ALLOWED_URLS_LIST, modelStore, "../mnist.mar");
    }

    @Test(
            expectedExceptions = ModelNotFoundException.class,
            expectedExceptionsMessageRegExp = "Model not found at: noop1.mar")
    public void testMarFileNotexist() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(ALLOWED_URLS_LIST, modelStore, "noop1.mar");
    }

    @Test(expectedExceptions = FileAlreadyExistsException.class)
    public void testFileAlreadyExist() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(
                ALLOWED_URLS_LIST,
                modelStore,
                "https://torchserve.pytorch.org/mar_files/mnist.mar");
    }

    @Test(expectedExceptions = DownloadModelException.class)
    public void testMalformLocalURL() throws ModelException, IOException, InterruptedException {
        String modelStore = "src/test/resources/models";
        ModelArchive.downloadModel(
                ALLOWED_URLS_LIST, modelStore, "file:///" + modelStore + "/mnist1.mar");
    }
}
