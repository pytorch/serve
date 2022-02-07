package org.pytorch.serve.util;

import io.netty.handler.ssl.SslContext;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.pytorch.serve.TestUtils;
import org.pytorch.serve.metrics.Dimension;
import org.pytorch.serve.metrics.Metric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ConfigManagerTest {
    static {
        TestUtils.init();
    }

    private Metric createMetric(String metricName, String requestId) {
        List<Dimension> dimensions = new ArrayList<>();
        Metric metric = new Metric();
        metric.setMetricName(metricName);
        metric.setRequestId(requestId);
        metric.setUnit("Milliseconds");
        metric.setTimestamp("1542157988");
        Dimension dimension = new Dimension();
        dimension.setName("Level");
        dimension.setValue("Model");
        dimensions.add(dimension);
        metric.setDimensions(dimensions);
        return metric;
    }

    @SuppressWarnings("unchecked")
    private void modifyEnv(String key, String val) throws ReflectiveOperationException {
        if (System.getProperty("os.name").toLowerCase().indexOf("win") >= 0) {
            Class<?> processEnvironmentClass = Class.forName("java.lang.ProcessEnvironment");
            Field f = processEnvironmentClass.getDeclaredField("theCaseInsensitiveEnvironment");
            f.setAccessible(true);
            Map<String, String> cienv = (Map<String, String>) f.get(null);
            cienv.put(key, val);
        } else {
            Map<String, String> env = System.getenv();
            Field field = env.getClass().getDeclaredField("m");
            field.setAccessible(true);
            ((Map<String, String>) field.get(env)).put(key, val);
        }
    }

    @Test
    public void test() throws IOException, GeneralSecurityException, ReflectiveOperationException {
        modifyEnv("TS_DEFAULT_RESPONSE_TIMEOUT", "130");
        ConfigManager.Arguments args = new ConfigManager.Arguments();
        args.setModels(new String[] {"noop_v0.1"});
        ConfigManager.init(args);
        ConfigManager configManager = ConfigManager.getInstance();
        configManager.setProperty("keystore", "src/test/resources/keystore.p12");
        Assert.assertEquals("true", configManager.getEnableEnvVarsConfig());
        Assert.assertEquals(130, configManager.getDefaultResponseTimeout());

        Dimension dimension;
        List<Metric> metrics = new ArrayList<>();
        // Create two metrics and add them to a list

        metrics.add(createMetric("TestMetric1", "12345"));
        metrics.add(createMetric("TestMetric2", "23478"));
        Logger logger = LoggerFactory.getLogger(ConfigManager.MODEL_SERVER_METRICS_LOGGER);
        logger.debug("{}", metrics);
        Assert.assertTrue(new File("build/logs/ts_metrics.log").exists());

        logger = LoggerFactory.getLogger(ConfigManager.MODEL_METRICS_LOGGER);
        logger.debug("{}", metrics);
        Assert.assertTrue(new File("build/logs/model_metrics.log").exists());

        Logger modelLogger = LoggerFactory.getLogger(ConfigManager.MODEL_LOGGER);
        modelLogger.debug("test model_log");
        Assert.assertTrue(new File("build/logs/model_log.log").exists());

        SslContext ctx = configManager.getSslContext();
        Assert.assertNotNull(ctx);
    }

    @Test
    public void testNoEnvVars() throws ReflectiveOperationException, IOException {
        System.setProperty("tsConfigFile", "src/test/resources/config_test_env.properties");
        modifyEnv("TS_DEFAULT_RESPONSE_TIMEOUT", "130");
        ConfigManager.Arguments args = new ConfigManager.Arguments();
        args.setModels(new String[] {"noop_v0.1"});
        args.setSnapshotDisabled(true);
        ConfigManager.init(args);
        ConfigManager configManager = ConfigManager.getInstance();
        Assert.assertEquals("false", configManager.getEnableEnvVarsConfig());
        Assert.assertEquals(120, configManager.getDefaultResponseTimeout());
        Assert.assertEquals(4, configManager.getJsonIntValue("noop", "1.0", "batchSize", 1));
        Assert.assertEquals(4, configManager.getJsonIntValue("vgg16", "1.0", "maxWorkers", 1));
        modifyEnv("TS_DEFAULT_RESPONSE_TIMEOUT", "120");
    }
}
