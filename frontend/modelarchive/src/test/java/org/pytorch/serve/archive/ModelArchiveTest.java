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

    @Test
    public void test() throws ModelException, IOException {
        String modelStore = "src/test/resources/models";

        // load 1.0 model from model folder
        ModelArchive archive = ModelArchive.downloadModel(modelStore, "noop-v1.0");
        Assert.assertEquals(archive.getModelName(), "noop");

        // load 1.0 model from model archive
        File src = new File(modelStore, "noop-v1.0");
        File target = new File("build/tmp/test", "noop-v1.0.mar");
        ZipUtils.zip(src, target, false);
        archive = ModelArchive.downloadModel("build/tmp/test", "noop-v1.0.mar");
        Assert.assertEquals(archive.getModelName(), "noop");

        // load model for s3
        archive =
                ModelArchive.downloadModel(
                        modelStore,
                        "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model");
        Assert.assertEquals(archive.getModelName(), "squeezenet_v1.1");

        // test export
        String[] args = new String[4];
        args[0] = "export";
        args[1] = "--model-name=noop";
        args[2] = "--model-path=" + archive.getModelDir();
        args[3] = "--output-file=" + output.getAbsolutePath();

        Exporter.main(args);
        Assert.assertTrue(output.exists());

        // load 1.0 model
        archive = ModelArchive.downloadModel(modelStore, "noop-v1.0");
        Assert.assertEquals(archive.getModelName(), "noop");
    }
}
