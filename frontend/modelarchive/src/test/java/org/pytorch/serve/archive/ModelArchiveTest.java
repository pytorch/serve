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

    @Test
    public void testInvalidModelVersionNull() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";

        ModelArchive archive = ModelArchive.downloadModel(modelStore, "noop.mar");
        archive.getManifest().getModel().setModelVersion(null);
        try{
            archive.validate();
        } catch (Exception e) {
            System.out.println("ModelVersion");
        }

        archive.getManifest().getModel().setModelName(null);
        try{
            archive.validate();
        } catch (Exception e) {
            System.out.println("ModelName");
        }

        archive.getManifest().setModel(null);
        try{
            archive.validate();
        } catch (Exception e) {
            System.out.println("Model");
        }

        archive.getManifest().setRuntime(null);
        try{
            archive.validate();
        } catch (Exception e) {
            System.out.println("Runtime");
        }

        archive.getManifest().setRuntime(null);
        try{
            archive.validate();
        } catch (Exception e) {
            System.out.println("Runtime");
        }

        try{
            archive.downloadModel(null, "/noop");
        } catch (Exception e) {
            System.out.println("ModelNotFound");
        }

        try{
            archive.downloadModel(null, "../noop");
        } catch (Exception e) {
            System.out.println("ModelNotFound");
        }

        try{
            archive.downloadModel("null", "/noop");
        } catch (Exception e) {
            System.out.println("ModelNotFound");
        }

        try{
            String handler = archive.getHandler();
        } catch (Exception e) {
            System.out.println("Handler");
        }

        try{
            String url = archive.getUrl();
        } catch (Exception e) {
            System.out.println("Url");
        }

        try{
            File file = archive.getModelDir();
        } catch (Exception e) {
            System.out.println("modelDir");
        }

        try{
            String name = archive.getModelName();
        } catch (Exception e) {
            System.out.println("modelName");
        }

        try{
            String version = archive.getModelVersion();
        } catch (Exception e) {
            System.out.println("modelVersion");
        }

        try{
            archive.readFile(new File("null"), Manifest.class);
        } catch (Exception e) {
            System.out.println("readFile");
        }

        archive.clean();
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
