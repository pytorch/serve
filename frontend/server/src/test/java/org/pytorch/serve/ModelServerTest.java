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

        try (InputStream is = new FileInputStream("src/test/resources/inference_open_api.json")) {
            listInferenceApisResult = IOUtils.toString(is, StandardCharsets.UTF_8.name());
        }

        try (InputStream is = new FileInputStream("src/test/resources/management_open_api.json")) {
            listManagementApisResult = IOUtils.toString(is, StandardCharsets.UTF_8.name());
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
    public void test()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException, NoSuchFieldException, IllegalAccessException {
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

        testPing(channel);

        testRoot(channel, listInferenceApisResult);
        testRoot(managementChannel, listManagementApisResult);
        testApiDescription(channel, listInferenceApisResult);
        testDescribeApi(channel);
        testUnregisterModel(managementChannel, "noop", null);
        testLoadModel(managementChannel, "noop.mar", "noop_v1.0", "1.11");
        testSyncScaleModel(managementChannel, "noop_v1.0", null);
        testSyncScaleModelWithoutVersion(managementChannel, "noop_v1.0");
        testListModels(managementChannel);
        testDescribeModel(managementChannel, "noop_v1.0", null, "1.11");
        testLoadModelWithInitialWorkers(managementChannel, "noop.mar", "noop", "1.11");
        testLoadModelWithInitialWorkers(managementChannel, "noop.mar", "noopversioned", "1.11");
        testLoadModelWithInitialWorkers(managementChannel, "noop_v2.mar", "noopversioned", "1.2.1");
        testDescribeModel(managementChannel, "noopversioned", null, "1.11");
        testDescribeModel(managementChannel, "noopversioned", "all", "1.2.1");
        testDescribeModel(managementChannel, "noopversioned", "1.11", "1.11");
        testPredictions(channel, "noopversioned", "OK", "1.2.1");
        testSetDefault(managementChannel, "noopversioned", "1.2.1");
        testLoadModelWithInitialWorkersWithJSONReqBody(managementChannel);
        testScaleModel(managementChannel);
        testPredictions(channel, "noop", "OK", null);
        testPredictionsBinary(channel);
        testPredictionsJson(channel);
        testInvocationsJson(channel);
        testInvocationsMultipart(channel);
        testModelsInvokeJson(channel);
        testModelsInvokeMultipart(channel);
        testLegacyPredict(channel);
        testPredictionsInvalidRequestSize(channel);
        testPredictionsValidRequestSize(channel);
        testPredictionsDecodeRequest(channel, managementChannel);
        testPredictionsDoNotDecodeRequest(channel, managementChannel);
        testPredictionsModifyResponseHeader(channel, managementChannel);
        testPredictionsNoManifest(channel, managementChannel);
        testModelRegisterWithDefaultWorkers(managementChannel);
        testLoadModelFromURL(managementChannel);
        testUnregisterURLModel(managementChannel);
        testLoadingMemoryError();
        testPredictionMemoryError();
        testMetricManager();
        testErrorBatch();

        channel.close();
        managementChannel.close();

        // negative test case, channel will be closed by server
        testInvalidRootRequest();
        testInvalidInferenceUri();
        testInvalidPredictionsUri();
        testInvalidDescribeModel();
        testPredictionsModelNotFound();

        testInvalidManagementUri();
        testInvalidModelsMethod();
        testInvalidModelMethod();
        testDescribeModelNotFound();
        testRegisterModelMissingUrl();
        testRegisterModelInvalidRuntime();
        testRegisterModelNotFound();
        testRegisterModelConflict();
        testRegisterModelMalformedUrl();
        testRegisterModelConnectionFailed();
        testRegisterModelHttpError();
        testRegisterModelInvalidPath();
        testScaleModelNotFound();
        testScaleModelFailure();
        testUnregisterModelNotFound();
        testUnregisterModelTimeout();
        testSetInvalidVersionDefault("noopversioned", "3.3.3");
        testUnregisterModelFailure("noopversioned", "1.2.1");

        testTS();
    }

    public void testTS()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException, NoSuchFieldException, IllegalAccessException {
        Channel channel = null;
        Channel managementChannel = null;
        for (int i = 0; i < 5; ++i) {
            channel = TestUtils.connect(false, configManager, 300);
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }

        for (int i = 0; i < 5; ++i) {
            managementChannel = TestUtils.connect(true, configManager, 300);
            if (managementChannel != null) {
                break;
            }
            Thread.sleep(100);
        }

        Assert.assertNotNull(channel, "Failed to connect to inference port.");
        Assert.assertNotNull(managementChannel, "Failed to connect to management port.");

        testLoadModelWithInitialWorkers(managementChannel, "mnist.mar", "mnist", "1.0");
        testPredictions(channel, "mnist", "0", null);
        testUnregisterModel(managementChannel, "mnist", null);
        testLoadModelWithInitialWorkers(
                managementChannel, "mnist_scripted.mar", "mnist_scripted", "1.0");
        testPredictions(channel, "mnist_scripted", "0", null);
        testUnregisterModel(managementChannel, "mnist_scripted", null);
        testLoadModelWithInitialWorkers(
                managementChannel, "mnist_traced.mar", "mnist_traced", "1.0");
        testPredictions(channel, "mnist_traced", "0", null);
        testUnregisterModel(managementChannel, "mnist_traced", null);

        channel.close();
        managementChannel.close();
    }

    private void testRoot(Channel channel, String expected) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.getRoot(channel);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getResult(), expected);
    }

    private void testPing(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/ping");
        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Healthy");
        Assert.assertTrue(TestUtils.getHeaders().contains("x-request-id"));
    }

    private void testApiDescription(Channel channel, String expected) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.getApiDescription(channel);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getResult(), expected);
    }

    private void testDescribeApi(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeModelApi(channel, "noop");
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getResult(), noopApiResult);
    }

    private void testLoadModel(Channel channel, String url, String modelName, String version)
            throws InterruptedException {
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

    private void testLoadModelFromURL(Channel channel) throws InterruptedException {
        testLoadModel(
                channel,
                "https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar",
                "squeezenet",
                "1.0");
        Assert.assertTrue(new File(configManager.getModelStore(), "squeezenet1_1.mar").exists());
    }

    private void testUnregisterURLModel(Channel channel) throws InterruptedException {
        testUnregisterModel(channel, "squeezenet", null);
        Assert.assertTrue(!new File(configManager.getModelStore(), "squeezenet1_1.mar").exists());
    }

    private void testLoadModelWithInitialWorkers(
            Channel channel, String url, String modelName, String version)
            throws InterruptedException {

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

    private void testLoadModelWithInitialWorkersWithJSONReqBody(Channel channel)
            throws InterruptedException {
        testUnregisterModel(channel, "noop", null);
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

    private void testScaleModel(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.scaleModel(channel, "noop_v1.0", null, 2, false);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Processing worker updates...");
    }

    private void testSyncScaleModel(Channel channel, String modelName, String version)
            throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.scaleModel(channel, modelName, version, 1, true);

        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "1 Workers scaled for model " + modelName);
    }

    private void testSyncScaleModelWithoutVersion(Channel channel, String modelName)
            throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.scaleModel(channel, modelName, null, 1, true);

        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "1 Workers scaled for model " + modelName);
    }

    private void testUnregisterModel(Channel channel, String modelName, String version)
            throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(channel, modelName, version, false);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Model \"" + modelName + "\" unregistered");
    }

    private void testUnregisterModelFailure(String modelName, String version)
            throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterModel(channel, modelName, version, false);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
        Assert.assertEquals(
                resp.getMessage(), "Cannot remove default version for model " + modelName);

        channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        testUnregisterModel(channel, "noopversioned", "1.11");
        testUnregisterModel(channel, "noopversioned", "1.2.1");
    }

    private void testListModels(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.listModels(channel);
        TestUtils.getLatch().await();

        ListModelsResponse resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), ListModelsResponse.class);
        Assert.assertEquals(resp.getModels().size(), 1);
    }

    private void testDescribeModel(
            Channel channel, String modelName, String requestVersion, String expectedVersion)
            throws InterruptedException {
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

    private void testSetDefault(Channel channel, String modelName, String defaultVersion)
            throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/models/" + modelName + "/" + defaultVersion + "/set-default";

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(
                resp.getStatus(),
                "Default vesion succsesfully updated for model \""
                        + modelName
                        + "\" to \""
                        + defaultVersion
                        + "\"");
    }

    private void testSetInvalidVersionDefault(String modelName, String defaultVersion)
            throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/models/" + modelName + "/" + defaultVersion + "/set-default";

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
        Assert.assertEquals(
                resp.getMessage(),
                "Model version " + defaultVersion + " does not exist for model " + modelName);
    }

    private void testPredictions(
            Channel channel, String modelName, String expectedOutput, String version)
            throws InterruptedException {
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

    private void testPredictionsJson(Channel channel) throws InterruptedException {
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

    private void testPredictionsBinary(Channel channel) throws InterruptedException {
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

    private void testInvocationsJson(Channel channel) throws InterruptedException {
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

    private void testInvocationsMultipart(Channel channel)
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
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

    private void testModelsInvokeJson(Channel channel) throws InterruptedException {
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

    private void testModelsInvokeMultipart(Channel channel)
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
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

    private void testPredictionsInvalidRequestSize(Channel channel) throws InterruptedException {
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

    private void testPredictionsValidRequestSize(Channel channel) throws InterruptedException {
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

    private void testModelRegisterWithDefaultWorkers(Channel mgmtChannel)
            throws NoSuchFieldException, IllegalAccessException, InterruptedException {
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

    private void testPredictionsDecodeRequest(Channel inferChannel, Channel mgmtChannel)
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
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

    private void testPredictionsDoNotDecodeRequest(Channel inferChannel, Channel mgmtChannel)
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
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

    private void testPredictionsModifyResponseHeader(
            Channel inferChannel, Channel managementChannel)
            throws NoSuchFieldException, IllegalAccessException, InterruptedException {
        setConfiguration("decode_input_request", "false");
        loadTests(managementChannel, "respheader-test.mar", "respheader");

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
        unloadTests(managementChannel, "respheader");
    }

    private void testPredictionsNoManifest(Channel inferChannel, Channel mgmtChannel)
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
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

    private void testLegacyPredict(Channel channel) throws InterruptedException {
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/noop/predict?data=test");
        channel.writeAndFlush(req);

        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getResult(), "OK");
    }

    private void testInvalidRootRequest() throws InterruptedException {
        Channel channel = TestUtils.connect(false, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    private void testInvalidInferenceUri() throws InterruptedException {
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

    private void testInvalidDescribeModel() throws InterruptedException {
        Channel channel = TestUtils.connect(false, configManager);
        Assert.assertNotNull(channel);

        TestUtils.describeModelApi(channel, "InvalidModel");
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    private void testInvalidPredictionsUri() throws InterruptedException {
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

    private void testPredictionsModelNotFound() throws InterruptedException {
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

    private void testInvalidManagementUri() throws InterruptedException {
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

    private void testInvalidModelsMethod() throws InterruptedException {
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

    private void testInvalidModelMethod() throws InterruptedException {
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

    private void testDescribeModelNotFound() throws InterruptedException {
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

    private void testRegisterModelMissingUrl() throws InterruptedException {
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

    private void testRegisterModelInvalidRuntime() throws InterruptedException {
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

    private void testRegisterModelNotFound() throws InterruptedException {
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

    private void testRegisterModelConflict() throws InterruptedException {
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

    private void testRegisterModelMalformedUrl() throws InterruptedException {
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

    private void testRegisterModelConnectionFailed() throws InterruptedException {
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

    private void testRegisterModelHttpError() throws InterruptedException {
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

    private void testRegisterModelInvalidPath() throws InterruptedException {
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

    private void testScaleModelNotFound() throws InterruptedException {
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

    private void testUnregisterModelNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);

        TestUtils.unregisterModel(channel, "fake", null, true);

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: fake");
    }

    private void testUnregisterModelTimeout()
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

    private void testScaleModelFailure() throws InterruptedException {
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

    private void testLoadingMemoryError() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerModel(channel, "loading-memory-error.mar", "memory_error", true, false);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.INSUFFICIENT_STORAGE);
        channel.close();
    }

    private void testPredictionMemoryError() throws InterruptedException {
        // Load the model
        Channel channel = TestUtils.connect(true, configManager);
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

    private void testErrorBatch() throws InterruptedException {
        Channel channel = TestUtils.connect(true, configManager);
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

    private void testMetricManager() throws JsonParseException, InterruptedException {
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
}
