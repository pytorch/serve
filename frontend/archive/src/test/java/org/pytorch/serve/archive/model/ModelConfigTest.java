package org.pytorch.serve.archive.model;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import org.pytorch.serve.archive.utils.ArchiveUtils;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ModelConfigTest {
    @Test
    public void TestValidYamlConfig() throws InvalidModelException, IOException {
        String yamlConfigFile = "src/test/resources/modelConfig/valid.yaml";
        ModelConfig modelConfig;
        File configFile = new File(yamlConfigFile);
        Map<String, Object> modelConfigMap = ArchiveUtils.readYamlFile(configFile);
        modelConfig = ModelConfig.build(modelConfigMap);

        Assert.assertEquals(modelConfig.getMinWorkers(), 1);
        Assert.assertEquals(modelConfig.getMaxWorkers(), 1);
        Assert.assertEquals(modelConfig.getBatchSize(), 1);
        Assert.assertEquals(modelConfig.getMaxBatchDelay(), 100);
        Assert.assertEquals(modelConfig.getResponseTimeout(), 120);
        Assert.assertEquals(modelConfig.getDeviceType(), ModelConfig.DeviceType.GPU);
        Assert.assertEquals(modelConfig.getParallelLevel(), 4);
        Assert.assertEquals(modelConfig.getParallelType(), ModelConfig.ParallelType.PP);
        Assert.assertEquals(modelConfig.getDeviceIds().get(2).intValue(), 2);
    }

    @Test
    public void TestInvalidYamlConfig() throws InvalidModelException, IOException {
        String yamlConfigFile = "src/test/resources/modelConfig/invalid.yaml";
        ModelConfig modelConfig;
        File configFile = new File(yamlConfigFile);
        Map<String, Object> modelConfigMap = ArchiveUtils.readYamlFile(configFile);
        modelConfig = ModelConfig.build(modelConfigMap);

        Assert.assertNotEquals(modelConfig.getMinWorkers(), 1);
        Assert.assertEquals(modelConfig.getMaxWorkers(), 1);
        Assert.assertEquals(modelConfig.getBatchSize(), 1);
        Assert.assertEquals(modelConfig.getMaxBatchDelay(), 100);
        Assert.assertEquals(modelConfig.getResponseTimeout(), 120);
        Assert.assertNotEquals(modelConfig.getDeviceType(), ModelConfig.DeviceType.GPU);
        Assert.assertEquals(modelConfig.getParallelLevel(), 4);
        Assert.assertNotEquals(modelConfig.getParallelType(), ModelConfig.ParallelType.PPTP);
        Assert.assertNull(modelConfig.getDeviceIds());
    }
}
