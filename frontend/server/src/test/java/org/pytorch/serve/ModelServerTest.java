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
import io.netty.handler.codec.http.multipart.HttpPostRequestEncoder;
import io.netty.handler.codec.http.multipart.MemoryFileUpload;
import io.netty.util.CharsetUtil;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.pytorch.serve.http.DescribeModelResponse;
import org.pytorch.serve.http.ErrorResponse;
import org.pytorch.serve.http.ListModelsResponse;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.metrics.Dimension;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.metrics.MetricManager;
import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.snapshot.InvalidSnapshotException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.JsonUtils;
import org.testng.Assert;
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
    private String noopApiResult;

    static {
        TestUtils.init();
    }

    @BeforeSuite
    public void beforeSuite()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        ConfigManager.init(new ConfigManager.Arguments());
        configManager = ConfigManager.getInstance();
        PluginsManager.getInstance().initialize();

        InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);

        server = new ModelServer(configManager);
        server.start();
        String version = configManager.getProperty("version", null);
        try (InputStream is = new FileInputStream("src/test/resources/inference_open_api.json")) {
            listInferenceApisResult =
                    String.format(IOUtils.toString(is, StandardCharsets.UTF_8.name()), version);
        }

        try (InputStream is = new FileInputStream("src/test/resources/management_open_api.json")) {
            listManagementApisResult =
                    String.format(IOUtils.toString(is, StandardCharsets.UTF_8.name()), version);
        }

        try (InputStream is = new FileInputStream("src/test/resources/describe_api.json")) {
            noopApiResult = IOUtils.toString(is, StandardCharsets.UTF_8.name());
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

        Assert.assertEquals(TestUtils.getResult(), listInferenceApisResult);
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

        Assert.assertEquals(TestUtils.getResult(), listManagementApisResult);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRootManagement"})
    public void testApiDescription() throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.getApiDescription(channel);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getResult(), listInferenceApisResult);
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

        Assert.assertEquals(TestUtils.getResult(), noopApiResult);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeApi"})
    public void testUnregisterNoopModel() throws InterruptedException {
        testUnregisterModel("noop", null);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterNoopModel"})
    public void testLoadNoopModel() throws InterruptedException {
        testLoadModel("noop.mar", "noop_v1.0", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadNoopModel"})
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
        testDescribeModel("noop_v1.0", null, "1.11");
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
        testDescribeModel("noopversioned", null, "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeDefaultModelVersion"})
    public void testDescribeAllModelVersion() throws InterruptedException {
        testDescribeModel("noopversioned", "all", "1.2.1");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeAllModelVersion"})
    public void testDescribeSpecificModelVersion() throws InterruptedException {
        testDescribeModel("noopversioned", "1.11", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeSpecificModelVersion"})
    public void testNoopVersionedPrediction() throws InterruptedException {
        testPredictions("noopversioned", "OK", "1.11");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testNoopVersionedPrediction"})
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
        setConfiguration("decode_input_request", "true");
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
    public void testPredictionsDoNotDecodeRequest()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        Channel inferChannel = TestUtils.getInferenceChannel(configManager);
        Channel mgmtChannel = TestUtils.getManagementChannel(configManager);
        setConfiguration("decode_input_request", "false");
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
    public void testPredictionsModifyResponseHeader()
            throws NoSuchFieldException, IllegalAccessException, InterruptedException {
        Channel inferChannel = TestUtils.getInferenceChannel(configManager);
        Channel mgmtChannel = TestUtils.getManagementChannel(configManager);
        setConfiguration("decode_input_request", "false");
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
        setConfiguration("default_service_handler", "service:handle");
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
        setConfiguration("default_workers_per_model", "1");
        loadTests(mgmtChannel, "noop.mar", "noop_default_model_workers");

        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.describeModel(mgmtChannel, "noop_default_model_workers", null);
        TestUtils.getLatch().await();

        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), DescribeModelResponse[].class);
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        Assert.assertEquals(resp[0].getMinWorkers(), 1);
        unloadTests(mgmtChannel, "noop_default_model_workers");
        setConfiguration("default_workers_per_model", "0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testModelRegisterWithDefaultWorkers"})
    public void testLoadModelFromURL() throws InterruptedException {
        testLoadModel(
                "https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar",
                "squeezenet",
                "1.0");
        Assert.assertTrue(new File(configManager.getModelStore(), "squeezenet1_1.mar").exists());
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadModelFromURL"})
    public void testUnregisterURLModel() throws InterruptedException {
        testUnregisterModel("squeezenet", null);
        Assert.assertTrue(!new File(configManager.getModelStore(), "squeezenet1_1.mar").exists());
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterURLModel"})
    public void testLoadingMemoryError() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerModel(channel, "loading-memory-error.mar", "memory_error", true, false);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.INSUFFICIENT_STORAGE);
        channel.close();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadingMemoryError"})
    public void testPredictionMemoryError() throws InterruptedException {
        // Load the model
        Channel channel = TestUtils.getManagementChannel(configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerModel(channel, "prediction-memory-error.mar", "pred-err", true, false);
        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
        channel.close();

        // Test for prediction
        channel = TestUtils.connect(false, configManager);
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
        channel.close();

        // Unload the model
        channel = TestUtils.connect(true, configManager);
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
    public void testErrorBatch() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
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

        channel.close();

        channel = TestUtils.connect(false, configManager);
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
        Channel channel = TestUtils.connect(false, configManager);
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
    public void testInvalidInferenceUri() throws InterruptedException {
        Channel channel = TestUtils.connect(false, configManager);
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
    public void testInvalidDescribeModel() throws InterruptedException {
        Channel channel = TestUtils.connect(false, configManager);
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
    public void testInvalidPredictionsUri() throws InterruptedException {
        Channel channel = TestUtils.connect(false, configManager);
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
        Channel channel = TestUtils.connect(false, configManager);
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
        Channel channel = TestUtils.connect(false, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
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
    public void testInvalidModelsMethod() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
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
    public void testDescribeModelVersionNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models?url=InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found in model store: InvalidUrl");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelNotFound"})
    public void testRegisterModelConflict() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
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
                resp.getMessage(), "Failed to download model from: http://localhost:aaaa");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelMalformedUrl"})
    public void testRegisterModelConnectionFailed() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
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
                "Failed to download model from: http://localhost:18888/fake.mar");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelConnectionFailed"})
    public void testRegisterModelHttpError() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
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
                "Failed to download model from: https://localhost:8443/fake.mar");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterModelHttpError"})
    public void testRegisterModelInvalidPath() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
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
    public void testUnregisterModelNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
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
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);

        TestUtils.unregisterModel(channel, "noopversioned", "1.3.1", true);

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
        Channel channel = TestUtils.connect(true, configManager);
        setConfiguration("unregister_model_timeout", "0");

        TestUtils.unregisterModel(channel, "noop_v1.0", null, true);

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.REQUEST_TIMEOUT.code());
        Assert.assertEquals(resp.getMessage(), "Timed out while cleaning resources: noop_v1.0");

        channel = TestUtils.connect(true, configManager);
        setConfiguration("unregister_model_timeout", "120");

        TestUtils.unregisterModel(channel, "noop_v1.0", null, true);
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterModelTimeout"})
    public void testScaleModelFailure() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
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
        Assert.assertEquals(resp.getMessage(), "Failed to start workers");
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
    public void testUnregisterModelFailure() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(channel, "noopversioned", "1.2.1", false);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.FORBIDDEN.code());
        Assert.assertEquals(
                resp.getMessage(), "Cannot remove default version for model noopversioned");

        channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.unregisterModel(channel, "noopversioned", "1.11", false);
        TestUtils.unregisterModel(channel, "noopversioned", "1.2.1", false);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterModelFailure"})
    public void testTSValidPort()
            throws InterruptedException, InvalidSnapshotException, GeneralSecurityException,
                    IOException {
        //  test case for verifying port range refer https://github.com/pytorch/serve/issues/291
        ConfigManager.init(new ConfigManager.Arguments());
        ConfigManager configManagerValidPort = ConfigManager.getInstance();
        FileUtils.deleteQuietly(new File(System.getProperty("LOG_LOCATION"), "config"));
        configManagerValidPort.setProperty("inference_address", "https://127.0.0.1:42523");
        ModelServer serverValidPort = new ModelServer(configManagerValidPort);
        serverValidPort.start();

        Channel channel = null;
        Channel managementChannel = null;
        for (int i = 0; i < 5; ++i) {
            channel = TestUtils.connect(false, configManagerValidPort);
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }

        for (int i = 0; i < 5; ++i) {
            managementChannel = TestUtils.connect(true, configManagerValidPort);
            if (managementChannel != null) {
                break;
            }
            Thread.sleep(100);
        }

        Assert.assertNotNull(channel, "Failed to connect to inference port.");
        Assert.assertNotNull(managementChannel, "Failed to connect to management port.");

        TestUtils.ping(configManagerValidPort);

        serverValidPort.stop();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testTSValidPort"})
    public void testTSInvalidPort()
            throws IOException, InterruptedException, GeneralSecurityException,
                    InvalidSnapshotException {
        //  test case for verifying port range refer https://github.com/pytorch/serve/issues/291
        //  invalid port test
        ConfigManager.init(new ConfigManager.Arguments());
        ConfigManager configManagerInvalidPort = ConfigManager.getInstance();
        FileUtils.deleteQuietly(new File(System.getProperty("LOG_LOCATION"), "config"));
        configManagerInvalidPort.setProperty("inference_address", "https://127.0.0.1:65536");
        ModelServer serverInvalidPort = new ModelServer(configManagerInvalidPort);
        try {
            serverInvalidPort.start();
        } catch (Exception e) {
            Assert.assertEquals(e.getClass(), IllegalArgumentException.class);
            Assert.assertEquals(e.getMessage(), "Invalid port number: https://127.0.0.1:65536");
        }
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

    private void testDescribeModel(String modelName, String requestVersion, String expectedVersion)
            throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModel(channel, modelName, requestVersion);
        TestUtils.getLatch().await();

        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), DescribeModelResponse[].class);
        if ("all".equals(requestVersion)) {
            Assert.assertTrue(resp.length >= 1);
        } else {
            Assert.assertTrue(resp.length == 1);
        }
        Assert.assertTrue(expectedVersion.equals(resp[0].getModelVersion()));
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

    private void setConfiguration(String key, String val)
            throws NoSuchFieldException, IllegalAccessException {
        Field f = configManager.getClass().getDeclaredField("prop");
        f.setAccessible(true);
        Properties p = (Properties) f.get(configManager);
        p.setProperty(key, val);
    }
}
