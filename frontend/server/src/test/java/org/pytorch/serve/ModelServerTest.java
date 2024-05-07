package org.pytorch.serve;

import com.google.gson.JsonParseException;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestEncoder;
import io.netty.handler.codec.http.multipart.MemoryAttribute;
import io.netty.handler.codec.http.multipart.MemoryFileUpload;
import io.netty.util.CharsetUtil;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.pytorch.serve.http.ErrorResponse;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.http.messages.DescribeModelResponse;
import org.pytorch.serve.http.messages.ListModelsResponse;
import org.pytorch.serve.metrics.Dimension;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.metrics.MetricManager;
import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.snapshot.InvalidSnapshotException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.wlm.Model;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

public class ModelServerTest {
    private static final String ERROR_NOT_FOUND =
            "Requested resource is not found, please refer to API document.";
    private static final String ERROR_METHOD_NOT_ALLOWED =
            "Requested method is not allowed, please refer to API document.";

    private ConfigManager configManager;
    private ModelServer server;
    private String listInferenceApisResult;
    private String listManagementApisResult;
    private String listMetricsApisResult;
    private String noopApiResult;
    private String noopManagementApiResult;

    static {
        TestUtils.init();
    }

    @BeforeSuite
    public void beforeSuite()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        ConfigManager.init(new ConfigManager.Arguments());
        configManager = ConfigManager.getInstance();
        configManager.setProperty("metrics_mode", "prometheus");
        PluginsManager.getInstance().initialize();
        MetricCache.init();

        InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);

        server = new ModelServer(configManager);
        server.startRESTserver();
        String version = configManager.getProperty("version", null);
        try (InputStream is = new FileInputStream("src/test/resources/inference_open_api.json")) {
            listInferenceApisResult =
                    String.format(IOUtils.toString(is, StandardCharsets.UTF_8.name()), version);
        }

        try (InputStream is = new FileInputStream("src/test/resources/management_open_api.json")) {
            listManagementApisResult =
                    String.format(IOUtils.toString(is, StandardCharsets.UTF_8.name()), version);
        }

        try (InputStream is = new FileInputStream("src/test/resources/metrics_open_api.json")) {
            listMetricsApisResult =
                    String.format(IOUtils.toString(is, StandardCharsets.UTF_8.name()), version);
        }

        try (InputStream is = new FileInputStream("src/test/resources/describe_api.json")) {
            noopApiResult = IOUtils.toString(is, StandardCharsets.UTF_8.name());
        }

        try (InputStream is = new FileInputStream("src/test/resources/model_management_api.json")) {
            noopManagementApiResult =
                    String.format(IOUtils.toString(is, StandardCharsets.UTF_8.name()), version);
        }
    }

    @AfterClass
    public void afterSuite() {
        server.stop();
    }

    @Test
    public void testPing() throws InterruptedException {
        TestUtils.ping(configManager);
        TestUtils.getLatch().await();
        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Healthy");
        Assert.assertTrue(TestUtils.getHeaders().contains("x-request-id"));
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPing"})
    public void testRootInference() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.getRoot(channel);
        TestUtils.getLatch().await();

        Assert.assertEquals(
                TestUtils.getResult().replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"),
                listInferenceApisResult.replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"));
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRootInference"})
    public void testRootManagement() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.getRoot(channel);
        TestUtils.getLatch().await();

        Assert.assertEquals(
                TestUtils.getResult().replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"),
                listManagementApisResult.replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"));
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRootManagement"})
    public void testRootMetrics() throws InterruptedException {
        Channel channel = TestUtils.getMetricsChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.getRoot(channel);
        TestUtils.getLatch().await();

        Assert.assertEquals(
                TestUtils.getResult().replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"),
                listMetricsApisResult.replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"));
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRootMetrics"})
    public void testApiDescription() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.getApiDescription(channel);
        TestUtils.getLatch().await();

        Assert.assertEquals(
                TestUtils.getResult().replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"),
                listInferenceApisResult.replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"));
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testApiDescription"})
    public void testDescribeApi() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModelApi(channel, "noop");
        TestUtils.getLatch().await();

        Assert.assertEquals(
                TestUtils.getResult().replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"),
                noopApiResult.replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"));
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeApi"})
    public void testModelManagementApi() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModelManagementApi(channel, "noop");
        TestUtils.getLatch().await();

        Assert.assertEquals(
                TestUtils.getResult().replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"),
                noopManagementApiResult.replaceAll("(\\\\r|\r\n|\n|\n\r)", "\r"));
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testModelManagementApi"})
    public void testInitialWorkers() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModel(channel, "noop", null, false);
        TestUtils.getLatch().await();
        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), DescribeModelResponse[].class);
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertEquals(
                resp[0].getMinWorkers(),
                configManager.getJsonIntValue(
                        "noop", "1.11", Model.MIN_WORKERS, configManager.getDefaultWorkers()));
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInitialWorkers"})
    public void testUnregisterNoopModel() throws InterruptedException {
        testUnregisterModel("noop", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterNoopModel"})
    public void testLoadNoopModel() throws InterruptedException {
        testLoadModel("noop.mar", "noop_v1.0", "1.11");
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModel(channel, "noop_v1.0", null, false);
        TestUtils.getLatch().await();
        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), DescribeModelResponse[].class);
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertEquals(
                resp[0].getMinWorkers(), configManager.getConfiguredDefaultWorkersPerModel());
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadNoopModel"})
    public void testDescribeModelCustomizedNoWorker() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/noop_v1.0?customized=true");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.SERVICE_UNAVAILABLE.code());
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeModelCustomizedNoWorker"})
    public void testSyncScaleNoopModel() throws InterruptedException {
        testSyncScaleModel("noop_v1.0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadNoopModel"})
    public void testSyncScaleNoopModelWithVersion() throws InterruptedException {
        testSyncScaleModel("noop_v1.0", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testSyncScaleNoopModel"})
    public void testListModels() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.listModels(channel);
        TestUtils.getLatch().await();

        ListModelsResponse resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), ListModelsResponse.class);
        Assert.assertEquals(resp.getModels().size(), 1);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testListModels"})
    public void testDescribeNoopModel() throws InterruptedException {
        testDescribeModel("noop_v1.0", null, false, "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeNoopModel"})
    public void testLoadNoopModelWithInitialWorkers() throws InterruptedException {
        testLoadModelWithInitialWorkers("noop.mar", "noop", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadNoopModelWithInitialWorkers"})
    public void testLoadNoopV1ModelWithInitialWorkers() throws InterruptedException {
        testLoadModelWithInitialWorkers("noop.mar", "noopversioned", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadNoopV1ModelWithInitialWorkers"})
    public void testLoadNoopV2ModelWithInitialWorkers() throws InterruptedException {
        testLoadModelWithInitialWorkers("noop_v2.mar", "noopversioned", "1.2.1");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadNoopV2ModelWithInitialWorkers"})
    public void testDescribeDefaultModelVersion() throws InterruptedException {
        testDescribeModel("noopversioned", null, false, "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeDefaultModelVersion"})
    public void testDescribeAllModelVersion() throws InterruptedException {
        testDescribeModel("noopversioned", "all", false, "1.2.1");
        testDescribeModel("noopversioned", "all", true, "1.2.1");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeAllModelVersion"})
    public void testDescribeSpecificModelVersion() throws InterruptedException {
        testDescribeModel("noopversioned", "1.11", false, "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeSpecificModelVersion"})
    public void testDescribeModelJobQueueStatus() throws InterruptedException {
        testLoadModelWithInitialWorkers("noop.mar", "noop_describe", "1.11");

        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModel(channel, "noop_describe", "1.11", false);
        TestUtils.getLatch().await();

        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), DescribeModelResponse[].class);
        Assert.assertEquals(resp[0].getJobQueueStatus().getRemainingCapacity(), 100);
        Assert.assertEquals(resp[0].getJobQueueStatus().getPendingRequests(), 0);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeModelJobQueueStatus"})
    public void testNoopVersionedPrediction() throws InterruptedException {
        testPredictions("noopversioned", "OK", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopVersionedPrediction"})
    public void testNoopVersionedExplanation() throws InterruptedException {
        testExplanations("noopversioned", "OK", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopVersionedExplanation"})
    public void testNoopVersionedKFV1Prediction() throws InterruptedException {
        testKFV1Predictions("noopversioned", "OK", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopVersionedKFV1Prediction"})
    public void testNoopVersionedKFV1Explanation() throws InterruptedException {
        testKFV1Explanations("noopversioned", "OK", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopVersionedExplanation"})
    public void testNoopVersionedKFV2Prediction() throws InterruptedException {
        testKFV2Predictions("noopversioned", "OK", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopVersionedKFV1Prediction"})
    public void testNoopVersionedKFV2Explanation() throws InterruptedException {
        testKFV2Explanations("noopversioned", "OK", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopVersionedKFV1Prediction"})
    public void testSetDefaultVersionNoop() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.setDefault(channel, "noopversioned", "1.2.1");
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(
                resp.getStatus(),
                "Default vesion succsesfully updated for model \"noopversioned\" to \"1.2.1\"");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testSetDefaultVersionNoop"})
    public void testLoadModelWithInitialWorkersWithJSONReqBody() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        testUnregisterModel("noop", null);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        req.headers().add("Content-Type", "application/json");
        req.content()
                .writeCharSequence(
                        "{'url':'noop.mar', 'model_name':'noop', 'initial_workers':'1', 'synchronous':'true'}",
                        CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(
                resp.getStatus(), "Model \"noop\" Version: 1.11 registered with 1 initial workers");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadModelWithInitialWorkersWithJSONReqBody"})
    public void testNoopPrediction() throws InterruptedException {
        testPredictions("noop", "OK", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopPrediction"})
    public void testLoadModelWithNakedDirNoVersionModelArchive() throws InterruptedException {
        String operatingSystem = System.getProperty("os.name").toLowerCase();
        if (!operatingSystem.contains("win")) {
            Channel channel = TestUtils.getManagementChannel(configManager);
            testUnregisterModel("noop", null);
            TestUtils.setResult(null);
            TestUtils.setLatch(new CountDownLatch(1));
            DefaultFullHttpRequest req =
                    new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
            req.headers().add("Content-Type", "application/json");
            req.content()
                    .writeCharSequence(
                            "{'url':'"
                                    + configManager.getModelStore()
                                    + "/noop_no_archive_no_version/noop/"
                                    + "', 'model_name':'noop', 'initial_workers':'1', 'synchronous':'true'}",
                            CharsetUtil.UTF_8);
            HttpUtil.setContentLength(req, req.content().readableBytes());
            channel.writeAndFlush(req);
            TestUtils.getLatch().await();

            StatusResponse resp =
                    JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
            Assert.assertEquals(
                    resp.getStatus(),
                    "Model \"noop\" Version: 1.0 registered with 1 initial workers");
        }
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadModelWithNakedDirNoVersionModelArchive"})
    public void testLoadModelWithNakedDirModelArchive() throws InterruptedException {
        String operatingSystem = System.getProperty("os.name").toLowerCase();
        if (!operatingSystem.contains("win")) {
            Channel channel = TestUtils.getManagementChannel(configManager);
            testUnregisterModel("noop", "1.0");
            TestUtils.setResult(null);
            TestUtils.setLatch(new CountDownLatch(1));
            DefaultFullHttpRequest req =
                    new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
            req.headers().add("Content-Type", "application/json");
            req.content()
                    .writeCharSequence(
                            "{'url':'"
                                    + configManager.getModelStore()
                                    + "/noop_no_archive/noop/"
                                    + "', 'model_name':'noop', 'initial_workers':'1', 'synchronous':'true'}",
                            CharsetUtil.UTF_8);
            HttpUtil.setContentLength(req, req.content().readableBytes());
            channel.writeAndFlush(req);
            TestUtils.getLatch().await();

            StatusResponse resp =
                    JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
            Assert.assertEquals(
                    resp.getStatus(),
                    "Model \"noop\" Version: 1.11 registered with 1 initial workers");
        }
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadModelWithNakedDirModelArchive"})
    public void testNoopExplanation() throws InterruptedException {
        testExplanations("noop", "OK", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopExplanation"})
    public void testNoopKFV1Prediction() throws InterruptedException {
        testKFV1Predictions("noop", "OK", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopKFV1Prediction"})
    public void testNoopKFV1Explanation() throws InterruptedException {
        testKFV1Explanations("noop", "OK", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopExplanation"})
    public void testNoopKFV2Prediction() throws InterruptedException {
        testKFV2Predictions("noop", "OK", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopKFV1Prediction"})
    public void testNoopKFV2Explanation() throws InterruptedException {
        testKFV2Explanations("noop", "OK", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopKFV1Explanation"})
    public void testPredictionsBinary() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");
        req.content().writeCharSequence("test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getResult(), "OK");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsBinary"})
    public void testPredictionsJson() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getResult(), "OK");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsJson"})
    public void testLoadModelWithHandlerName() throws InterruptedException {
        testLoadModelWithInitialWorkers("noop_handlername.mar", "noop_handlername", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadModelWithHandlerName"})
    public void testNoopWithHandlerNamePrediction() throws InterruptedException {
        testPredictions("noop_handlername", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithHandlerNamePrediction"})
    public void testNoopWithHandlerNameExplanation() throws InterruptedException {
        testExplanations("noop_handlername", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithHandlerNameExplanation"})
    public void testNoopWithHandlerNameKFV1Prediction() throws InterruptedException {
        testKFV1Predictions("noop_handlername", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithHandlerNameKFV1Prediction"})
    public void testNoopWithHandlerNameKFV1Explanation() throws InterruptedException {
        testKFV1Explanations("noop_handlername", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithHandlerNameExplanation"})
    public void testNoopWithHandlerNameKFV2Prediction() throws InterruptedException {
        testKFV2Predictions("noop_handlername", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithHandlerNameKFV1Prediction"})
    public void testNoopWithHandlerNameKFV2Explanation() throws InterruptedException {
        testKFV2Explanations("noop_handlername", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithHandlerNameKFV1Explanation"})
    public void testLoadModelWithEntryPntFuncName() throws InterruptedException {
        testLoadModelWithInitialWorkers("noop_entrypntfunc.mar", "noop_entrypntfunc", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadModelWithEntryPntFuncName"})
    public void testNoopWithEntryPntFuncPrediction() throws InterruptedException {
        testPredictions("noop_entrypntfunc", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithEntryPntFuncPrediction"})
    public void testNoopWithEntryPntFuncExplanation() throws InterruptedException {
        testExplanations("noop_entrypntfunc", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithEntryPntFuncExplanation"})
    public void testNoopWithEntryPntFuncKFV1Prediction() throws InterruptedException {
        testKFV1Predictions("noop_entrypntfunc", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithEntryPntFuncKFV1Prediction"})
    public void testNoopWithEntryPntFuncKFV1Explanation() throws InterruptedException {
        testKFV1Explanations("noop_entrypntfunc", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithEntryPntFuncExplanation"})
    public void testNoopWithEntryPntFuncKFV2Prediction() throws InterruptedException {
        testKFV2Predictions("noop_entrypntfunc", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithEntryPntFuncKFV1Prediction"})
    public void testNoopWithEntryPntFuncKFV2Explanation() throws InterruptedException {
        testKFV2Explanations("noop_entrypntfunc", "OK", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopWithEntryPntFuncKFV1Explanation"})
    public void testInvocationsJson() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations?model_name=noop");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getResult(), "OK");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInvocationsJson"})
    public void testInvocationsMultipart()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations");

        HttpPostRequestEncoder encoder = new HttpPostRequestEncoder(req, true);
        encoder.addBodyAttribute("model_name", "noop_v1.0");
        MemoryFileUpload body =
                new MemoryFileUpload("data", "test.txt", "text/plain", null, null, 4);
        body.setContent(Unpooled.copiedBuffer("test", StandardCharsets.UTF_8));
        encoder.addBodyHttpData(body);

        channel.writeAndFlush(encoder.finalizeRequest());
        if (encoder.isChunked()) {
            channel.writeAndFlush(encoder).sync();
        }

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getResult(), "OK");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInvocationsMultipart"})
    public void testModelsInvokeJson() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop/invoke");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getResult(), "OK");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testModelsInvokeJson"})
    public void testModelsInvokeMultipart()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop/invoke");

        HttpPostRequestEncoder encoder = new HttpPostRequestEncoder(req, true);
        MemoryFileUpload body =
                new MemoryFileUpload("data", "test.txt", "text/plain", null, null, 4);
        body.setContent(Unpooled.copiedBuffer("test", StandardCharsets.UTF_8));
        encoder.addBodyHttpData(body);

        channel.writeAndFlush(encoder.finalizeRequest());
        if (encoder.isChunked()) {
            channel.writeAndFlush(encoder).sync();
        }

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getResult(), "OK");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testModelsInvokeMultipart"})
    public void testLegacyPredict() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/noop/predict?data=test");
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getResult(), "OK");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLegacyPredict"})
    public void testPredictionsInvalidRequestSize() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");

        req.content().writeZero(11485760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.REQUEST_ENTITY_TOO_LARGE);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsInvalidRequestSize"})
    public void testPredictionsValidRequestSize() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");

        req.content().writeZero(10385760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsValidRequestSize"})
    public void testPredictionsDecodeRequest()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        Channel inferChannel = TestUtils.getInferenceChannel(configManager);
        Channel mgmtChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setConfiguration(configManager, "decode_input_request", "true");
        loadTests(mgmtChannel, "noop-v1.0-config-tests.mar", "noop-config");
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop-config");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertFalse(TestUtils.getResult().contains("bytearray"));
        unloadTests(mgmtChannel, "noop-config");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsDecodeRequest"})
    public void testNoopCustomized() throws InterruptedException {
        testLoadModelWithInitialWorkers("noop-customized.mar", "noop-customized", "1.0");

        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModel(channel, "noop-customized", "1.0", true);
        TestUtils.getLatch().await();
        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), DescribeModelResponse[].class);
        Assert.assertEquals(
                resp[0].getCustomizedMetadata().toString(), "{\"data1\":\"1\",\"data2\":\"2\"}");

        testUnregisterModel("noop-customized", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopCustomized"})
    public void testPredictionsDoNotDecodeRequest()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        Channel inferChannel = TestUtils.getInferenceChannel(configManager);
        Channel mgmtChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setConfiguration(configManager, "decode_input_request", "false");
        loadTests(mgmtChannel, "noop-v1.0-config-tests.mar", "noop-config");

        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop-config");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertTrue(TestUtils.getResult().contains("bytearray"));
        unloadTests(mgmtChannel, "noop-config");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsDoNotDecodeRequest"})
    public void testPredictionsEchoMultipart()
            throws HttpPostRequestEncoder.ErrorDataEncoderException, InterruptedException,
                    IOException {
        Channel inferChannel = TestUtils.getInferenceChannel(configManager);
        Channel mgmtChannel = TestUtils.getManagementChannel(configManager);
        loadTests(mgmtChannel, "echo.mar", "echo");

        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/echo");

        ByteBuffer allBytes = ByteBuffer.allocate(0x100);
        IntStream.range(0, 0x100).forEach(i -> allBytes.put((byte) i));

        HttpPostRequestEncoder encoder = new HttpPostRequestEncoder(req, true);
        MemoryFileUpload data =
                new MemoryFileUpload(
                        "data", "allBytes.bin", "application/octet-stream", null, null, 0x100);
        data.setContent(Unpooled.copiedBuffer(allBytes));
        encoder.addBodyHttpData(data);

        inferChannel.writeAndFlush(encoder.finalizeRequest());
        if (encoder.isChunked()) {
            inferChannel.writeAndFlush(encoder).sync();
        }

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertEquals(TestUtils.getContent(), data.get());
        unloadTests(mgmtChannel, "echo");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsEchoMultipart"})
    public void testPredictionsEchoNoMultipart()
            throws HttpPostRequestEncoder.ErrorDataEncoderException, InterruptedException,
                    IOException {
        Channel inferChannel = TestUtils.getInferenceChannel(configManager);
        Channel mgmtChannel = TestUtils.getManagementChannel(configManager);
        loadTests(mgmtChannel, "echo.mar", "echo");

        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/echo");

        ByteBuffer allBytes = ByteBuffer.allocate(0x100);
        IntStream.range(0, 0x100).forEach(i -> allBytes.put((byte) i));

        Charset charset = StandardCharsets.ISO_8859_1;
        HttpPostRequestEncoder.EncoderMode mode = HttpPostRequestEncoder.EncoderMode.RFC1738;
        HttpPostRequestEncoder encoder =
                new HttpPostRequestEncoder(new DefaultHttpDataFactory(), req, false, charset, mode);
        MemoryAttribute data = new MemoryAttribute("data", charset);
        data.setContent(Unpooled.copiedBuffer(allBytes));
        encoder.addBodyHttpData(data);

        inferChannel.writeAndFlush(encoder.finalizeRequest());
        if (encoder.isChunked()) {
            inferChannel.writeAndFlush(encoder).sync();
        }

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertEquals(TestUtils.getContent(), data.get());
        unloadTests(mgmtChannel, "echo");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsEchoNoMultipart"})
    public void testPredictionsModifyResponseHeader()
            throws NoSuchFieldException, IllegalAccessException, InterruptedException {
        Channel inferChannel = TestUtils.getInferenceChannel(configManager);
        Channel mgmtChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setConfiguration(configManager, "decode_input_request", "false");
        loadTests(mgmtChannel, "respheader-test.mar", "respheader");

        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/respheader");

        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertEquals(TestUtils.getHeaders().get("dummy"), "1");
        Assert.assertEquals(TestUtils.getHeaders().get("content-type"), "text/plain");
        Assert.assertTrue(TestUtils.getResult().contains("bytearray"));
        unloadTests(mgmtChannel, "respheader");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsModifyResponseHeader"})
    public void testPredictionsNoManifest()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        Channel inferChannel = TestUtils.getInferenceChannel(configManager);
        Channel mgmtChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setConfiguration(configManager, "default_service_handler", "service:handle");
        loadTests(mgmtChannel, "noop-no-manifest.mar", "nomanifest");
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/nomanifest");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertEquals(TestUtils.getResult(), "OK");
        unloadTests(mgmtChannel, "nomanifest");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsNoManifest"})
    public void testModelRegisterWithDefaultWorkers()
            throws NoSuchFieldException, IllegalAccessException, InterruptedException {
        Channel mgmtChannel = TestUtils.getManagementChannel(configManager);
        TestUtils.setConfiguration(configManager, "default_workers_per_model", "1");
        loadTests(mgmtChannel, "noop.mar", "noop_default_model_workers");

        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.describeModel(mgmtChannel, "noop_default_model_workers", null, false);
        TestUtils.getLatch().await();

        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), DescribeModelResponse[].class);
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertEquals(resp[0].getMinWorkers(), 1);
        unloadTests(mgmtChannel, "noop_default_model_workers");
        TestUtils.setConfiguration(configManager, "default_workers_per_model", "0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testModelRegisterWithDefaultWorkers"})
    public void testLoadModelFromURL() throws InterruptedException {
        testLoadModel(
                "https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar", "squeezenet", "1.0");
        Assert.assertTrue(new File(configManager.getModelStore(), "squeezenet1_1.mar").exists());
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadModelFromURL"})
    public void testLoadModelFromFileURI() throws InterruptedException, IOException {
        String curDir = System.getProperty("user.dir");
        File curDirFile = new File(curDir);
        String parent = curDirFile.getParent();

        String source = configManager.getModelStore() + "/mnist.mar";
        String destination = parent + "/archive/mnist1.mar";
        File sourceFile = new File(source);
        File destinationFile = new File(destination);
        String fileUrl = "";
        FileUtils.copyFile(sourceFile, destinationFile);
        fileUrl = "file:///" + parent + "/archive/mnist1.mar";
        testLoadModel(fileUrl, "mnist1", "1.0");
        Assert.assertTrue(new File(configManager.getModelStore(), "mnist1.mar").exists());
        FileUtils.deleteQuietly(destinationFile);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadModelFromFileURI"})
    public void testUnregisterFileURIModel() throws InterruptedException {
        testUnregisterModel("mnist1", null);
        Assert.assertFalse(new File(configManager.getModelStore(), "mnist1.mar").exists());
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterFileURIModel"})
    public void testModelWithInvalidFileURI() throws InterruptedException, IOException {
        String invalidFileUrl = "file:///InvalidUrl";
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        TestUtils.setHttpStatus(null);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(channel, invalidFileUrl, "invalid_file_url", true, false);
        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.BAD_REQUEST);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testModelWithInvalidFileURI"})
    public void testUnregisterURLModel() throws InterruptedException {
        testUnregisterModel("squeezenet", null);
        Assert.assertFalse(new File(configManager.getModelStore(), "squeezenet1_1.mar").exists());
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterURLModel"})
    public void testModelWithCustomPythonDependency()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        TestUtils.setConfiguration(configManager, "install_py_dep_per_model", "true");
        testLoadModelWithInitialWorkers("custom_python_dep.mar", "custom_python_dep", "1.0");
        TestUtils.setConfiguration(configManager, "install_py_dep_per_model", "false");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testModelWithCustomPythonDependency"})
    public void testModelWithInvalidCustomPythonDependency()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        TestUtils.setConfiguration(configManager, "install_py_dep_per_model", "true");
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(
                channel,
                "custom_invalid_python_dep.mar",
                "custom_invalid_python_dep",
                false,
                false);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.BAD_REQUEST);
        Assert.assertEquals(
                resp.getMessage(),
                "Custom pip package installation failed for model custom_invalid_python_dep");
        TestUtils.setConfiguration(configManager, "install_py_dep_per_model", "false");
        channel.close().sync();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testModelWithInvalidCustomPythonDependency"})
    public void testLoadingMemoryError() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }

        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerModel(channel, "loading-memory-error.mar", "memory_error", true, false);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.INSUFFICIENT_STORAGE);
        channel.close().sync();
        channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(channel, "memory_error", null, false);
        TestUtils.getLatch().await();
        channel.close().sync();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadingMemoryError"})
    public void testPredictionMemoryError() throws InterruptedException {
        // Load the model
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerModel(channel, "prediction-memory-error.mar", "pred-err", true, false);
        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        channel.close().sync();

        // Test for prediction
        channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/pred-err");
        req.content().writeCharSequence("data=invalid_output", CharsetUtil.UTF_8);

        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.INSUFFICIENT_STORAGE);
        channel.close().sync();

        // Unload the model
        channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        TestUtils.setHttpStatus(null);
        TestUtils.setLatch(new CountDownLatch(1));
        Assert.assertNotNull(channel);

        TestUtils.unregisterModel(channel, "pred-err", null, false);
        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionMemoryError"})
    public void testPredictionCustomErrorCode() throws InterruptedException {
        // Load the model
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerModel(
                channel, "pred-custom-return-code.mar", "pred-custom-return-code", true, false);
        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        channel.close().sync();

        // Test for prediction
        channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/predictions/pred-custom-return-code");
        req.content().writeCharSequence("data=invalid_output", CharsetUtil.UTF_8);

        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus().code(), 599);
        channel.close().sync();

        // Unload the model
        channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        TestUtils.setHttpStatus(null);
        TestUtils.setLatch(new CountDownLatch(1));
        Assert.assertNotNull(channel);

        TestUtils.unregisterModel(channel, "pred-custom-return-code", null, false);
        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionCustomErrorCode"})
    public void testErrorBatch() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        TestUtils.setHttpStatus(null);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerModel(channel, "error_batch.mar", "err_batch", true, false);
        TestUtils.getLatch().await();

        StatusResponse status =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(
                status.getStatus(),
                "Model \"err_batch\" Version: 1.0 registered with 1 initial workers");

        channel.close().sync();

        channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.setHttpStatus(null);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/err_batch");
        req.content().writeCharSequence("data=invalid_output", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.INSUFFICIENT_STORAGE);
        Assert.assertEquals(TestUtils.getResult(), "Invalid response");
        channel.close().sync();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testErrorBatch"})
    public void testMetricManager() throws JsonParseException, InterruptedException {
        MetricManager.scheduleMetrics(configManager);
        MetricManager metricManager = MetricManager.getInstance();
        List<Metric> metrics = metricManager.getMetrics();

        // Wait till first value is read in
        int count = 0;
        while (metrics.isEmpty()) {
            Thread.sleep(500);
            metrics = metricManager.getMetrics();
            Assert.assertTrue(++count < 5);
        }
        for (Metric metric : metrics) {
            if (metric.getMetricName().equals("CPUUtilization")) {
                Assert.assertEquals(metric.getUnit(), "Percent");
            }
            if (metric.getMetricName().equals("MemoryUsed")) {
                Assert.assertEquals(metric.getUnit(), "Megabytes");
            }
            if (metric.getMetricName().equals("DiskUsed")) {
                List<Dimension> dimensions = metric.getDimensions();
                for (Dimension dimension : dimensions) {
                    if (dimension.getName().equals("Level")) {
                        Assert.assertEquals(dimension.getValue(), "Host");
                    }
                }
            }
        }
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testMetricManager"})
    public void testInvalidRootRequest() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInvalidRootRequest"})
    public void testInvalidInferenceUri() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInvalidInferenceUri"})
    public void testInvalidDescribeModel() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        TestUtils.describeModelApi(channel, "InvalidModel");
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInvalidDescribeModel"})
    public void testInvalidPredictionsUri() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInvalidPredictionsUri"})
    public void testPredictionsModelNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions/InvalidModel");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsModelNotFound"})
    public void testPredictionsModelVersionNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions/noopversioned/1.3.1");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(
                resp.getMessage(), "Model version: 1.3.1 does not exist for model: noopversioned");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionsModelNotFound"})
    public void testInvalidManagementUri() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInvalidManagementUri"})
    public void testInvalidModelsMethod() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInvalidModelsMethod"})
    public void testInvalidModelMethod() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testInvalidModelMethod"})
    public void testDescribeModelNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/InvalidModel");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeModelNotFound"})
    public void testDescribeModelVersionNotFound() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/noopversioned/1.3.1");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(
                resp.getMessage(), "Model version: 1.3.1 does not exist for model: noopversioned");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeModelNotFound"})
    public void testRegisterModelMissingUrl() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(resp.getMessage(), "Parameter url is required.");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelMissingUrl"})
    public void testRegisterModelInvalidRuntime() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=InvalidUrl&runtime=InvalidRuntime");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(resp.getMessage(), "Invalid RuntimeType value: InvalidRuntime");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelInvalidRuntime"})
    public void testRegisterModelNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models?url=InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelNotFound"})
    public void testRegisterModelConflict() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop.mar&model_name=noop_v1.0&runtime=python&synchronous=false");
        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop.mar&model_name=noop_v1.0&runtime=python&synchronous=false");
        channel.writeAndFlush(req);
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.CONFLICT.code());
        Assert.assertEquals(
                resp.getMessage(), "Model version 1.11 is already registered for model noop_v1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelConflict"})
    public void testRegisterModelMalformedUrl() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=http%3A%2F%2Flocalhost%3Aaaaa");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(), "Failed to download archive from: http://localhost:aaaa");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelMalformedUrl"})
    public void testRegisterModelConnectionFailed() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=http%3A%2F%2Flocalhost%3A18888%2Ffake.mar&synchronous=false");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(),
                "Failed to download archive from: http://localhost:18888/fake.mar");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelConnectionFailed"})
    public void testRegisterModelHttpError() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=https%3A%2F%2Flocalhost%3A8443%2Ffake.mar&synchronous=false");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(),
                "Failed to download archive from: https://localhost:8443/fake.mar");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelHttpError"})
    public void testRegisterModelInvalidPath() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=..%2Ffake.mar&synchronous=false");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Relative path is not allowed in url: ../fake.mar");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelInvalidPath"})
    public void testScaleModelNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models/fake");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: fake");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testScaleModelNotFound"})
    public void testScaleModelVersionNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.scaleModel(channel, "noop_v1.0", "1.3.1", 2, true);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(
                resp.getMessage(), "Model version: 1.3.1 does not exist for model: noop_v1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testScaleModelNotFound"})
    public void testUnregisterModelNotFound() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        TestUtils.unregisterModel(channel, "fake", null, true);

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: fake");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterModelNotFound"})
    public void testUnregisterModelVersionNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.unregisterModel(channel, "noopversioned", "1.3.1", false);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(
                resp.getMessage(), "Model version: 1.3.1 does not exist for model: noopversioned");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterModelNotFound"})
    public void testUnregisterModelTimeout()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.setConfiguration(configManager, "unregister_model_timeout", "0");

        TestUtils.unregisterModel(channel, "noop_v1.0", null, false);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.REQUEST_TIMEOUT.code());
        Assert.assertEquals(resp.getMessage(), "Timed out while cleaning resources: noop_v1.0");

        channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.setConfiguration(configManager, "unregister_model_timeout", "120");

        TestUtils.unregisterModel(channel, "noop_v1.0", null, false);
        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterModelTimeout"})
    public void testScaleModelFailure() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }

        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        TestUtils.setHttpStatus(null);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerModel(channel, "init-error.mar", "init-error", false, false);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);

        TestUtils.setHttpStatus(null);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.scaleModel(channel, "init-error", null, 1, true);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.INTERNAL_SERVER_ERROR);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
        Assert.assertEquals(
                resp.getMessage(), "Failed to start workers for model init-error version: null");

        TestUtils.ping(configManager);
        TestUtils.getLatch().await();
        // There is a retry time window. To reduce CI latency,
        // it is fine for ping to either 200 or 500.
        Assert.assertTrue(
                TestUtils.getHttpStatus().equals(HttpResponseStatus.INTERNAL_SERVER_ERROR)
                        || TestUtils.getHttpStatus().equals(HttpResponseStatus.OK));
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testScaleModelFailure"})
    public void testLoadMNISTEagerModel() throws InterruptedException {
        testLoadModelWithInitialWorkers("mnist.mar", "mnist", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadMNISTEagerModel"})
    public void testPredictionMNISTEagerModel() throws InterruptedException {
        testPredictions("mnist", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionMNISTEagerModel"})
    public void testExplanationMNISTEagerModel() throws InterruptedException {
        testExplanations("mnist", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testExplanationMNISTEagerModel"})
    public void testKFV1PredictionMNISTEagerModel() throws InterruptedException {
        testKFV1Predictions("mnist", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testKFV1PredictionMNISTEagerModel"})
    public void testKFV1ExplanationMNISTEagerModel() throws InterruptedException {
        testKFV1Explanations("mnist", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testExplanationMNISTEagerModel"})
    public void testKFV2PredictionMNISTEagerModel() throws InterruptedException {
        testKFV2Predictions("mnist", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testKFV1PredictionMNISTEagerModel"})
    public void testKFV2ExplanationMNISTEagerModel() throws InterruptedException {
        testKFV2Explanations("mnist", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testKFV1ExplanationMNISTEagerModel"})
    public void testUnregistedMNISTEagerModel() throws InterruptedException {
        testUnregisterModel("mnist", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregistedMNISTEagerModel"})
    public void testLoadMNISTScriptedModel() throws InterruptedException {
        testLoadModelWithInitialWorkers("mnist_scripted.mar", "mnist_scripted", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadMNISTScriptedModel"})
    public void testPredictionMNISTScriptedModel() throws InterruptedException {
        testPredictions("mnist_scripted", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionMNISTScriptedModel"})
    public void testExplanationMNISTScriptedModel() throws InterruptedException {
        testExplanations("mnist_scripted", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testExplanationMNISTScriptedModel"})
    public void testKFV1PredictionMNISTScriptedModel() throws InterruptedException {
        testKFV1Predictions("mnist_scripted", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testKFV1PredictionMNISTScriptedModel"})
    public void testKFV1ExplanationMNISTScriptedModel() throws InterruptedException {
        testKFV1Explanations("mnist_scripted", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testExplanationMNISTScriptedModel"})
    public void testKFV2PredictionMNISTScriptedModel() throws InterruptedException {
        testKFV2Predictions("mnist_scripted", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testKFV1PredictionMNISTScriptedModel"})
    public void testKFV2ExplanationMNISTScriptedModel() throws InterruptedException {
        testKFV2Explanations("mnist_scripted", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testKFV1ExplanationMNISTScriptedModel"})
    public void testUnregistedMNISTScriptedModel() throws InterruptedException {
        testUnregisterModel("mnist_scripted", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregistedMNISTScriptedModel"})
    public void testLoadMNISTTracedModel() throws InterruptedException {
        testLoadModelWithInitialWorkers("mnist_traced.mar", "mnist_traced", "1.0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadMNISTTracedModel"})
    public void testPredictionMNISTTracedModel() throws InterruptedException {
        testPredictions("mnist_traced", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionMNISTTracedModel"})
    public void testExplanationMNISTTracedModel() throws InterruptedException {
        testExplanations("mnist_traced", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testExplanationMNISTTracedModel"})
    public void testKFV1PredictionMNISTTracedModel() throws InterruptedException {
        testKFV1Predictions("mnist_traced", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testKFV1PredictionMNISTTracedModel"})
    public void testKFV1ExplanationMNISTTracedModel() throws InterruptedException {
        testKFV1Explanations("mnist_traced", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testExplanationMNISTTracedModel"})
    public void testKFV2PredictionMNISTTracedModel() throws InterruptedException {
        testKFV2Predictions("mnist_traced", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testKFV1PredictionMNISTTracedModel"})
    public void testKFV2ExplanationMNISTTracedModel() throws InterruptedException {
        testKFV2Explanations("mnist_traced", "0", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testKFV1ExplanationMNISTTracedModel"})
    public void testUnregistedMNISTTracedModel() throws InterruptedException {
        testUnregisterModel("mnist_traced", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregistedMNISTTracedModel"})
    public void testSetInvalidDefaultVersion() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.setDefault(channel, "noopversioned", "3.3.3");
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(
                resp.getMessage(), "Model version 3.3.3 does not exist for model noopversioned");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testSetInvalidDefaultVersion"})
    public void testUnregisterModelFailure() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(channel, "noopversioned", "1.2.1", false);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.FORBIDDEN.code());
        Assert.assertEquals(
                resp.getMessage(), "Cannot remove default version for model noopversioned");

        channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(channel, "noopversioned", "1.11", false);
        TestUtils.getLatch().await();

        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(channel, "noopversioned", "1.2.1", false);
        TestUtils.getLatch().await();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterModelFailure"})
    public void testClientTimeout() throws InterruptedException {
        Channel mgmtChannel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        loadTests(mgmtChannel, "echo-client-timeout.mar", "echo-client-timeout");

        Channel inferChannel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/echo-client-timeout");
        req.content().writeZero(10385760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        inferChannel.writeAndFlush(req);
        TestUtils.getLatch().await(1, TimeUnit.SECONDS);
        Assert.assertNull(TestUtils.result);

        unloadTests(mgmtChannel, "echo-client-timeout");
    }

    private void testLoadModel(String url, String modelName, String version)
            throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(channel, url, modelName, false, false);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(
                resp.getStatus(),
                "Model \""
                        + modelName
                        + "\" Version: "
                        + version
                        + " registered with 0 initial workers. Use scale workers API to add workers for the model.");
    }

    private void testUnregisterModel(String modelName, String version) throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(channel, modelName, version, false);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Model \"" + modelName + "\" unregistered");
    }

    private void testSyncScaleModel(String modelName, String version) throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.scaleModel(channel, modelName, version, 1, true);

        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        if (version == null) {
            Assert.assertEquals(resp.getStatus(), "Workers scaled to 1 for model: " + modelName);
        } else {
            Assert.assertEquals(
                    resp.getStatus(),
                    "Workers scaled to 1 for model: " + modelName + ", version: " + version);
        }
    }

    private void testDescribeModel(
            String modelName, String requestVersion, boolean customized, String expectedVersion)
            throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModel(channel, modelName, requestVersion, customized);
        TestUtils.getLatch().await();

        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), DescribeModelResponse[].class);
        if ("all".equals(requestVersion)) {
            Assert.assertTrue(resp.length >= 1);
        } else {
            Assert.assertEquals(resp.length, 1);
        }
        Assert.assertEquals(resp[0].getModelVersion(), expectedVersion);
    }

    private void testLoadModelWithInitialWorkers(String url, String modelName, String version)
            throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(channel, url, modelName, true, false);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(
                resp.getStatus(),
                "Model \""
                        + modelName
                        + "\" Version: "
                        + version
                        + " registered with 1 initial workers");
    }

    private void testPredictions(String modelName, String expectedOutput, String version)
            throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/predictions/" + modelName;
        if (version != null) {
            requestURL += "/" + version;
        }
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        req.content().writeCharSequence("data=test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getResult(), expectedOutput);
        testModelMetrics(modelName, version);
    }

    private void testExplanations(String modelName, String expectedOutput, String version)
            throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/explanations/" + modelName;
        if (version != null) {
            requestURL += "/" + version;
        }
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        req.content().writeCharSequence("data=test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getResult(), expectedOutput);
        testModelMetrics(modelName, version);
    }

    private void testKFV1Predictions(String modelName, String expectedOutput, String version)
            throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/v1/models/" + modelName + ":predict";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        req.content().writeCharSequence("data=test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getResult(), expectedOutput);
        testModelMetrics(modelName, version);
    }

    private void testKFV1Explanations(String modelName, String expectedOutput, String version)
            throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/v1/models/" + modelName + ":explain";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        req.content().writeCharSequence("data=test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getResult(), expectedOutput);
        testModelMetrics(modelName, version);
    }

    private void testKFV2Predictions(String modelName, String expectedOutput, String version)
            throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/v2/models/" + modelName + "/infer";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        req.content().writeCharSequence("data=test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getResult(), expectedOutput);
        testModelMetrics(modelName, version);
    }

    private void testKFV2Explanations(String modelName, String expectedOutput, String version)
            throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/v2/models/" + modelName + "/explain";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        req.content().writeCharSequence("data=test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getResult(), expectedOutput);
        testModelMetrics(modelName, version);
    }

    private void testModelMetrics(String modelName, String version) throws InterruptedException {
        Channel metricsChannel = TestUtils.getMetricsChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest metricsReq =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/metrics");
        metricsChannel.writeAndFlush(metricsReq);
        TestUtils.getLatch().await();
        Pattern inferLatencyMatcher = TestUtils.getTSInferLatencyMatcher(modelName, version);
        Assert.assertTrue(inferLatencyMatcher.matcher(TestUtils.getResult()).find());

        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        metricsReq =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.GET,
                        "/metrics?name[]=ts_inference_latency_microseconds");
        metricsChannel.writeAndFlush(metricsReq);
        TestUtils.getLatch().await();
        Assert.assertTrue(inferLatencyMatcher.matcher(TestUtils.getResult()).find());
        Assert.assertFalse(TestUtils.getResult().contains("ts_inference_requests_total"));
    }

    private void loadTests(Channel channel, String model, String modelName)
            throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerModel(channel, model, modelName, true, false);

        TestUtils.getLatch().await();
    }

    private void unloadTests(Channel channel, String modelName) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String expected = "Model \"" + modelName + "\" unregistered";
        TestUtils.unregisterModel(channel, modelName, null, false);
        TestUtils.getLatch().await();
        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), expected);
    }
}
