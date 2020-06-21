package org.pytorch.serve;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import io.netty.channel.Channel;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.util.CharsetUtil;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.util.Comparator;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.snapshot.InvalidSnapshotException;
import org.pytorch.serve.snapshot.Snapshot;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class SnapshotTest {

    static {
        TestUtils.init();
    }

    private ConfigManager configManager;
    private ModelServer server;
    private String lastSnapshot;
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    @BeforeClass
    public void beforeSuite()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        System.setProperty("tsConfigFile", "src/test/resources/config.properties");
        FileUtils.deleteQuietly(new File(System.getProperty("LOG_LOCATION"), "config"));
        ConfigManager.init(new ConfigManager.Arguments());
        configManager = ConfigManager.getInstance();
        PluginsManager.getInstance().initialize();

        InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);
        server = new ModelServer(configManager);
        server.start();
    }

    @AfterClass
    public void afterSuite() throws InterruptedException {
        TestUtils.closeChannels();
        server.stop();
    }

    @Test
    public void testStartupSnapshot() {
        validateSnapshot("snapshot1.cfg");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testStartupSnapshot"})
    public void testUnregisterSnapshot() throws InterruptedException {
        Channel managementChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(managementChannel, "noop", null, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot2.cfg");
        waitForSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterSnapshot"})
    public void testRegisterSnapshot() throws InterruptedException {
        Channel managementChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(managementChannel, "noop.mar", "noop_v1.0", false, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot3.cfg");
        waitForSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterSnapshot"})
    public void testSyncScaleModelSnapshot() throws InterruptedException {
        Channel managementChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.scaleModel(managementChannel, "noop_v1.0", null, 1, true);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot4.cfg");
        waitForSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testSyncScaleModelSnapshot"})
    public void testNoSnapshotOnListModels() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.listModels(channel);
        TestUtils.getLatch().await();
        validateNoSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoSnapshotOnListModels"})
    public void testNoSnapshotOnDescribeModel() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModel(channel, "noop_v1.0", null);
        TestUtils.getLatch().await();
        validateNoSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoSnapshotOnDescribeModel"})
    public void testLoadModelWithInitialWorkersSnapshot() throws InterruptedException {
        Channel managementChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(managementChannel, "noop.mar", "noop", true, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot5.cfg");
        waitForSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadModelWithInitialWorkersSnapshot"})
    public void testNoSnapshotOnPrediction() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/predictions/noop_v1.0";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        req.content().writeCharSequence("data=test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoSnapshotOnPrediction"})
    public void testRegisterSecondModelSnapshot() throws InterruptedException {
        Channel managementChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(managementChannel, "noop.mar", "noopversioned", true, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot6.cfg");
        waitForSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterSecondModelSnapshot"})
    public void testSecondModelVersionSnapshot() throws InterruptedException {
        Channel managementChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(managementChannel, "noop_v2.mar", "noopversioned", true, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot7.cfg");
        waitForSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testSecondModelVersionSnapshot"})
    public void testSetDefaultSnapshot() throws InterruptedException {
        Channel managementChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/models/noopversioned/1.2.1/set-default";

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        managementChannel.writeAndFlush(req);
        TestUtils.getLatch().await();

        validateSnapshot("snapshot8.cfg");
        waitForSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testSetDefaultSnapshot"})
    public void testAsyncScaleModelSnapshot() throws InterruptedException {
        Channel managementChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.scaleModel(managementChannel, "noop_v1.0", null, 2, false);
        TestUtils.getLatch().await();
        waitForSnapshot(5000);
        validateSnapshot("snapshot9.cfg");
        waitForSnapshot();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testAsyncScaleModelSnapshot"})
    public void testStopTorchServeSnapshot() {
        server.stop();
        validateSnapshot("snapshot9.cfg");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testStopTorchServeSnapshot"})
    public void testStartTorchServeWithLastSnapshot()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        System.setProperty("tsConfigFile", "");
        ConfigManager.init(new ConfigManager.Arguments());
        configManager = ConfigManager.getInstance();
        server = new ModelServer(configManager);
        server.start();
        Channel channel = null;
        for (int i = 0; i < 5; ++i) {
            channel = TestUtils.connect(false, configManager);
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }
        validateSnapshot("snapshot9.cfg");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testStartTorchServeWithLastSnapshot"})
    public void testRestartTorchServeWithSnapshotAsConfig()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        server.stop();
        validateSnapshot("snapshot9.cfg");

        System.setProperty("tsConfigFile", getLastSnapshot());
        ConfigManager.init(new ConfigManager.Arguments());
        configManager = ConfigManager.getInstance();
        server = new ModelServer(configManager);
        server.start();
        Channel channel = null;
        for (int i = 0; i < 5; ++i) {
            channel = TestUtils.connect(false, configManager);
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }
        validateSnapshot("snapshot9.cfg");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRestartTorchServeWithSnapshotAsConfig"})
    public void testNoSnapshotOnInvalidModelRegister() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.registerModel(channel, "InvalidModel", "InvalidModel", false, true);

        validateSnapshot("snapshot9.cfg");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoSnapshotOnInvalidModelRegister"})
    public void testNoSnapshotOnInvalidModelUnregister() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.unregisterModel(channel, "InvalidModel", null, true);

        validateSnapshot("snapshot9.cfg");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoSnapshotOnInvalidModelUnregister"})
    public void testNoSnapshotOnInvalidModelVersionUnregister() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.registerModel(channel, "noopversioned", "3.0", false, true);

        validateSnapshot("snapshot9.cfg");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoSnapshotOnInvalidModelVersionUnregister"})
    public void testNoSnapshotOnInvalidModelScale() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.scaleModel(channel, "invalidModel", null, 1, true);

        validateSnapshot("snapshot9.cfg");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoSnapshotOnInvalidModelScale"})
    public void testNoSnapshotOnInvalidModelVersionScale() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.scaleModel(channel, "noopversioned", "3.0", 1, true);

        validateSnapshot("snapshot9.cfg");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoSnapshotOnInvalidModelVersionScale"})
    public void testNoSnapshotOnInvalidModelVersionSetDefault() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        String requestURL = "/models/noopversioned/3.0/set-default";

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        validateSnapshot("snapshot9.cfg");
    }

    private void validateNoSnapshot() {
        validateSnapshot(lastSnapshot);
    }

    private void validateSnapshot(String expectedSnapshot) {
        lastSnapshot = expectedSnapshot;
        File expectedSnapshotFile = new File("src/test/resources/snapshots", expectedSnapshot);
        Properties expectedProp = new Properties();

        try (InputStream stream = Files.newInputStream(expectedSnapshotFile.toPath())) {
            expectedProp.load(stream);
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read configuration file", e);
        }

        updateSnapshot(expectedProp);

        Properties actualProp = new Properties();
        File actualSnapshotFile = new File(getLastSnapshot());

        try (InputStream stream = Files.newInputStream(actualSnapshotFile.toPath())) {
            actualProp.load(stream);
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read configuration file", e);
        }

        updateSnapshot(actualProp);
        assert actualProp.equals(expectedProp);
    }

    private void updateSnapshot(Properties prop) {
        Snapshot snapshot = GSON.fromJson(prop.getProperty("model_snapshot"), Snapshot.class);
        snapshot.setName("snapshot");
        snapshot.setCreated(123456);
        for (Map.Entry<String, Map<String, JsonObject>> modelMap :
                snapshot.getModels().entrySet()) {
            for (Map.Entry<String, JsonObject> versionModel : modelMap.getValue().entrySet()) {
                versionModel.getValue().addProperty(Model.MIN_WORKERS, 4);
                versionModel.getValue().addProperty(Model.MAX_WORKERS, 4);
            }
        }
        String snapshotJson = GSON.toJson(snapshot, Snapshot.class);
        prop.put("model_snapshot", snapshotJson);
        prop.put("tsConfigFile", "dummyconfig");
        prop.put("NUM_WORKERS", 4);
        prop.put("number_of_gpu", 4);
        prop.put("version", "0.1.1");
    }

    private String getLastSnapshot() {
        String latestSnapshotPath = null;
        Path configPath = Paths.get(System.getProperty("LOG_LOCATION"), "config");

        if (Files.exists(configPath)) {
            try {
                Optional<Path> lastFilePath =
                        Files.list(configPath)
                                .filter(f -> !Files.isDirectory(f))
                                .max(Comparator.comparingLong(f -> f.toFile().lastModified()));
                if (lastFilePath.isPresent()) {
                    latestSnapshotPath = lastFilePath.get().toString();
                }
            } catch (IOException e) {
                e.printStackTrace(); // NOPMD
            }
        }

        return latestSnapshotPath;
    }

    private void waitForSnapshot(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void waitForSnapshot() {
        waitForSnapshot(1000);
    }
}
