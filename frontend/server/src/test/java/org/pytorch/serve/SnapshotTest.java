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
    public void afterSuite() {
        server.stop();
    }

    @Test
    public void test()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        Channel channel = null;
        Channel managementChannel = null;
        for (int i = 0; i < 5; ++i) {
            channel = TestUtils.connect(false, configManager);
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }

        for (int i = 0; i < 5; ++i) {
            managementChannel = TestUtils.connect(true, configManager);
            if (managementChannel != null) {
                break;
            }
            Thread.sleep(100);
        }

        Assert.assertNotNull(channel, "Failed to connect to inference port.");
        Assert.assertNotNull(managementChannel, "Failed to connect to management port.");

        testStartupSnapshot("snapshot1.cfg");
        testUnregisterSnapshot(managementChannel);
        testRegisterSnapshot(managementChannel);
        testSyncScaleModelSnapshot(managementChannel);
        testNoSnapshotOnListModels(managementChannel);
        testNoSnapshotOnDescribeModel(managementChannel);
        testLoadModelWithInitialWorkersSnapshot(managementChannel);
        testRegisterSecondModelSnapshot(managementChannel);
        testSecondModelVersionSnapshot(managementChannel);
        testNoSnapshotOnPrediction(channel);
        testSetDefaultSnapshot(managementChannel);
        testAsyncScaleModelSnapshot(managementChannel);

        channel.close();
        managementChannel.close();

        testStopTorchServeSnapshot();
        testStartTorchServeWithLastSnapshot();
        testRestartTorchServeWithSnapshotAsConfig();

        // Negative management API calls, channel will be closed by server
        testNoSnapshotOnInvalidModelRegister();
        testNoSnapshotOnInvalidModelUnregister();
        testNoSnapshotOnInvalidModelVersionUnregister();
        testNoSnapshotOnInvalidModelScale();
        testNoSnapshotOnInvalidModelVersionScale();
        testNoSnapshotOnInvalidModelVersionSetDefault();
    }

    private void testStartupSnapshot(String expectedSnapshot) {
        validateSnapshot(expectedSnapshot);
    }

    private void testUnregisterSnapshot(Channel managementChannel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(managementChannel, "noop", null, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot2.cfg");
        waitForSnapshot();
    }

    private void testRegisterSnapshot(Channel managementChannel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(managementChannel, "noop.mar", "noop_v1.0", false, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot3.cfg");
        waitForSnapshot();
    }

    private void testSyncScaleModelSnapshot(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.scaleModel(channel, "noop_v1.0", null, 1, true);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot4.cfg");
        waitForSnapshot();
    }

    private void testNoSnapshotOnListModels(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.listModels(channel);
        TestUtils.getLatch().await();
        validateNoSnapshot();
    }

    private void testNoSnapshotOnDescribeModel(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModel(channel, "noop_v1.0", null);
        TestUtils.getLatch().await();
        validateNoSnapshot();
    }

    private void testLoadModelWithInitialWorkersSnapshot(Channel channel)
            throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(channel, "noop.mar", "noop", true, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot5.cfg");
        waitForSnapshot();
    }

    private void testNoSnapshotOnPrediction(Channel channel) {
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

    private void testRegisterSecondModelSnapshot(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(channel, "noop.mar", "noopversioned", true, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot6.cfg");
        waitForSnapshot();
    }

    private void testSecondModelVersionSnapshot(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(channel, "noop_v2.mar", "noopversioned", true, false);
        TestUtils.getLatch().await();
        validateSnapshot("snapshot7.cfg");
        waitForSnapshot();
    }

    private void testSetDefaultSnapshot(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/models/noopversioned/1.2.1/set-default";

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        validateSnapshot("snapshot8.cfg");
        waitForSnapshot();
    }

    private void testAsyncScaleModelSnapshot(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.scaleModel(channel, "noop_v1.0", null, 2, false);
        TestUtils.getLatch().await();
        waitForSnapshot(5000);
        validateSnapshot("snapshot9.cfg");
        waitForSnapshot();
    }

    private void testStopTorchServeSnapshot() {
        server.stop();
        validateSnapshot("snapshot9.cfg");
    }

    private void testStartTorchServeWithLastSnapshot()
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

    private void testRestartTorchServeWithSnapshotAsConfig()
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

    private void validateNoSnapshot() {
        validateSnapshot(lastSnapshot);
    }

    private void testNoSnapshotOnInvalidModelRegister() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.registerModel(channel, "InvalidModel", "InvalidModel", false, true);

        validateSnapshot("snapshot9.cfg");
    }

    private void testNoSnapshotOnInvalidModelUnregister() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.unregisterModel(channel, "InvalidModel", null, true);

        validateSnapshot("snapshot9.cfg");
    }

    private void testNoSnapshotOnInvalidModelVersionUnregister() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.registerModel(channel, "noopversioned", "3.0", false, true);

        validateSnapshot("snapshot9.cfg");
    }

    private void testNoSnapshotOnInvalidModelScale() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.scaleModel(channel, "invalidModel", null, 1, true);

        validateSnapshot("snapshot9.cfg");
    }

    private void testNoSnapshotOnInvalidModelVersionScale() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.scaleModel(channel, "noopversioned", "3.0", 1, true);

        validateSnapshot("snapshot9.cfg");
    }

    private void testNoSnapshotOnInvalidModelVersionSetDefault() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        String requestURL = "/models/noopversioned/3.0/set-default";

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        validateSnapshot("snapshot9.cfg");
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
