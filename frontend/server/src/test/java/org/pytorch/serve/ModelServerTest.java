package org.pytorch.serve;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpContentDecompressor;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.multipart.HttpPostRequestEncoder;
import io.netty.handler.codec.http.multipart.MemoryFileUpload;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import io.netty.handler.stream.ChunkedWriteHandler;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.util.CharsetUtil;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.FileInputStream;
import java.io.FileReader;
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
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.pytorch.serve.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

public class ModelServerTest {

    private static final String ERROR_NOT_FOUND =
            "Requested resource is not found, please refer to API document.";
    private static final String ERROR_METHOD_NOT_ALLOWED =
            "Requested method is not allowed, please refer to API document.";

    private ConfigManager configManager;
    private ModelServer server;
    CountDownLatch latch;
    HttpResponseStatus httpStatus;
    String result;
    HttpHeaders headers;
    private String listInferenceApisResult;
    private String listManagementApisResult;
    private String noopApiResult;
    private String getCheckpointAPIResult;
    private String getAllCheckpointAPIResult;

    static {
        TestUtils.init();
    }

    @BeforeSuite
    public void beforeSuite() throws InterruptedException, IOException, GeneralSecurityException {
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

        JsonParser parser = new JsonParser();
        JsonArray allCheckpointJson =
                (JsonArray)
                        parser.parse(new FileReader("src/test/resources/all_checkpoints_api.json"));
        getAllCheckpointAPIResult = allCheckpointJson.toString();

        JsonArray checkpointJson =
                (JsonArray) parser.parse(new FileReader("src/test/resources/checkpoints_api.json"));
        getCheckpointAPIResult = checkpointJson.toString();
    }

    @AfterSuite
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
            channel = connect(false);
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }

        for (int i = 0; i < 5; ++i) {
            managementChannel = connect(true);
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
        testLoadModel(managementChannel, "noop.mar", "noop_v1.0");
        testSyncScaleModel(managementChannel, "noop_v1.0", null);
        testScaleModel(managementChannel);
        testListModels(managementChannel);
        testDescribeModel(managementChannel, "noop_v1.0", null, "1.11");
        testLoadModelWithInitialWorkers(managementChannel, "noop.mar", "noop");
        testLoadModelWithInitialWorkers(managementChannel, "noop.mar", "noopversioned");
        testLoadModelWithInitialWorkers(managementChannel, "noop_v2.mar", "noopversioned");
        testDescribeModel(managementChannel, "noopversioned", null, "1.21");
        testDescribeModel(managementChannel, "noopversioned", "all", "1.11");
        testDescribeModel(managementChannel, "noopversioned", "1.11", "1.11");
        testPredictions(channel, "noopversioned", "OK", "1.21");
        testSetDefault(managementChannel, "noopversioned", "1.11");
        testLoadModelWithInitialWorkersWithJSONReqBody(managementChannel);
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
        testUnregisterModelFailure("noopversioned", "1.11");
    }

    @Test
    public void testTS()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException, NoSuchFieldException, IllegalAccessException {
        Channel channel = null;
        Channel managementChannel = null;
        for (int i = 0; i < 5; ++i) {
            channel = connect(false);
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }

        for (int i = 0; i < 5; ++i) {
            managementChannel = connect(true);
            if (managementChannel != null) {
                break;
            }
            Thread.sleep(100);
        }

        Assert.assertNotNull(channel, "Failed to connect to inference port.");
        Assert.assertNotNull(managementChannel, "Failed to connect to management port.");

        testLoadModelWithInitialWorkers(managementChannel, "mnist.mar", "mnist");
        testPredictions(channel, "mnist", "0", null);
        testUnregisterModel(managementChannel, "mnist", null);
        testLoadModelWithInitialWorkers(managementChannel, "mnist_scripted.mar", "mnist_scripted");
        testPredictions(channel, "mnist_scripted", "0", null);
        testUnregisterModel(managementChannel, "mnist_scripted", null);
        testLoadModelWithInitialWorkers(managementChannel, "mnist_traced.mar", "mnist_traced");
        testPredictions(channel, "mnist_traced", "0", null);
        testUnregisterModel(managementChannel, "mnist_traced", null);

        channel.close();
        managementChannel.close();
    }

    @Test
    public void testCheckpointAPIs()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException, NoSuchFieldException, IllegalAccessException {
        Channel managementChannel = null;

        for (int i = 0; i < 5; ++i) {
            managementChannel = connect(true);
            if (managementChannel != null) {
                break;
            }
            Thread.sleep(100);
        }

        Assert.assertNotNull(managementChannel, "Failed to connect to management port.");

        testLoadModel(managementChannel, "noop.mar", "noop_v1.0");

        testCheckpoint(managementChannel);
        testGetCheckpoint(managementChannel);
        testGetAllCheckpoint(managementChannel);
        testRestoreCheckpoint(managementChannel);
        testRemoveCheckpoint(managementChannel);

        managementChannel.close();
    }

    private void testRoot(Channel channel, String expected) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/");
        channel.writeAndFlush(req).sync();
        latch.await();

        Assert.assertEquals(result, expected);
    }

    private void testPing(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/ping");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Healthy");
        Assert.assertTrue(headers.contains("x-request-id"));
    }

    private void testApiDescription(Channel channel, String expected) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/api-description");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, expected);
    }

    private void testDescribeApi(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/noop");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, noopApiResult);
    }

    private void testLoadModel(Channel channel, String url, String modelName)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL =
                "/models?url="
                        + url
                        + "&model_name="
                        + modelName
                        + "&runtime=python&synchronous=false";
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Model \"" + modelName + "\" registered");
    }

    private void testLoadModelWithInitialWorkers(Channel channel, String url, String modelName)
            throws InterruptedException {

        result = null;
        latch = new CountDownLatch(1);
        String requestURL =
                "/models?url="
                        + url
                        + "&model_name="
                        + modelName
                        + "&runtime=python&initial_workers=1&synchronous=true";
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Workers scaled");
    }

    private void testLoadModelWithInitialWorkersWithJSONReqBody(Channel channel)
            throws InterruptedException {
        testUnregisterModel(channel, "noop", null);

        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        req.headers().add("Content-Type", "application/json");
        req.content()
                .writeCharSequence(
                        "{'url':'noop.mar', 'model_name':'noop', 'initial_workers':'1', 'synchronous':'true'}",
                        CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Workers scaled");
    }

    private void testScaleModel(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models/noop_v1.0?min_worker=2");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Processing worker updates...");
    }

    private void testSyncScaleModel(Channel channel, String modelName, String version)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/models/" + modelName;
        if (version != null) {
            requestURL += "/" + version;
        }
        requestURL += "?synchronous=true&min_worker=1\"";
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Workers scaled");
    }

    private void testUnregisterModel(Channel channel, String modelName, String version)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/models/" + modelName;
        if (version != null) {
            requestURL += "/" + version;
        }

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, requestURL);
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Model \"" + modelName + "\" unregistered");
    }

    private void testUnregisterModelFailure(String modelName, String version)
            throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/models/" + modelName;
        if (version != null) {
            requestURL += "/" + version;
        }

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, requestURL);
        channel.writeAndFlush(req);
        latch.await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
        Assert.assertEquals(
                resp.getMessage(), "Cannot remove default version for model " + modelName);

        channel = connect(true);
        Assert.assertNotNull(channel);
        testUnregisterModel(channel, "noopversioned", "1.21");
        testUnregisterModel(channel, "noopversioned", "1.11");
    }

    private void testListModels(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models?limit=200&nextPageToken=X");
        channel.writeAndFlush(req);
        latch.await();

        ListModelsResponse resp = JsonUtils.GSON.fromJson(result, ListModelsResponse.class);
        Assert.assertEquals(resp.getModels().size(), 1);
    }

    private void testDescribeModel(
            Channel channel, String modelName, String requestVersion, String expectedVersion)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/models/" + modelName;
        if (requestVersion != null) {
            requestURL += "/" + requestVersion;
        }
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, requestURL);
        channel.writeAndFlush(req);
        latch.await();

        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(result, DescribeModelResponse[].class);
        if ("all".equals(requestVersion)) {
            Assert.assertTrue(resp.length >= 1);
        } else {
            Assert.assertTrue(resp.length == 1);
        }
        Assert.assertTrue(expectedVersion.equals(resp[0].getModelVersion()));
    }

    private void testSetDefault(Channel channel, String modelName, String defaultVersion)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/models/" + modelName + "/" + defaultVersion + "/set-default";

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(
                resp.getStatus(),
                "Default vesion succsesfully updated for model \""
                        + modelName
                        + "\" to \""
                        + defaultVersion
                        + "\"");
    }

    private void testPredictions(
            Channel channel, String modelName, String expectedOutput, String version)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
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

        latch.await();
        Assert.assertEquals(result, expectedOutput);
    }

    private void testPredictionsJson(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        channel.writeAndFlush(req);

        latch.await();
        Assert.assertEquals(result, "OK");
    }

    private void testPredictionsBinary(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");
        req.content().writeCharSequence("test", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testInvocationsJson(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations?model_name=noop");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testInvocationsMultipart(Channel channel)
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        result = null;
        latch = new CountDownLatch(1);
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

        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testModelsInvokeJson(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop/invoke");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testModelsInvokeMultipart(Channel channel)
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        result = null;
        latch = new CountDownLatch(1);
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

        latch.await();

        Assert.assertEquals(result, "OK");
    }

    private void testPredictionsInvalidRequestSize(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");

        req.content().writeZero(11485760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.REQUEST_ENTITY_TOO_LARGE);
    }

    private void testPredictionsValidRequestSize(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop");

        req.content().writeZero(10385760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
    }

    private void loadTests(Channel channel, String model, String modelName)
            throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String url =
                "/models?url="
                        + model
                        + "&model_name="
                        + modelName
                        + "&initial_workers=1&synchronous=true";
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        channel.writeAndFlush(req);
        latch.await();
    }

    private void unloadTests(Channel channel, String modelName) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String expected = "Model \"" + modelName + "\" unregistered";
        String url = "/models/" + modelName;
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, url);
        channel.writeAndFlush(req);
        latch.await();
        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
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

        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/noop_default_model_workers");
        mgmtChannel.writeAndFlush(req);

        latch.await();
        DescribeModelResponse[] resp =
                JsonUtils.GSON.fromJson(result, DescribeModelResponse[].class);
        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertEquals(resp[0].getMinWorkers(), 1);
        unloadTests(mgmtChannel, "noop_default_model_workers");
        setConfiguration("default_workers_per_model", "0");
    }

    private void testPredictionsDecodeRequest(Channel inferChannel, Channel mgmtChannel)
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        setConfiguration("decode_input_request", "true");
        loadTests(mgmtChannel, "noop-v1.0-config-tests.mar", "noop-config");

        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop-config");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertFalse(result.contains("bytearray"));
        unloadTests(mgmtChannel, "noop-config");
    }

    private void testPredictionsDoNotDecodeRequest(Channel inferChannel, Channel mgmtChannel)
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        setConfiguration("decode_input_request", "false");
        loadTests(mgmtChannel, "noop-v1.0-config-tests.mar", "noop-config");

        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/noop-config");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertTrue(result.contains("bytearray"));
        unloadTests(mgmtChannel, "noop-config");
    }

    private void testPredictionsModifyResponseHeader(
            Channel inferChannel, Channel managementChannel)
            throws NoSuchFieldException, IllegalAccessException, InterruptedException {
        setConfiguration("decode_input_request", "false");
        loadTests(managementChannel, "respheader-test.mar", "respheader");
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/respheader");

        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertEquals(headers.get("dummy"), "1");
        Assert.assertEquals(headers.get("content-type"), "text/plain");
        Assert.assertTrue(result.contains("bytearray"));
        unloadTests(managementChannel, "respheader");
    }

    private void testPredictionsNoManifest(Channel inferChannel, Channel mgmtChannel)
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        setConfiguration("default_service_handler", "service:handle");
        loadTests(mgmtChannel, "noop-no-manifest.mar", "nomanifest");

        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/nomanifest");
        req.content().writeCharSequence("{\"data\": \"test\"}", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        inferChannel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        Assert.assertEquals(result, "OK");
        unloadTests(mgmtChannel, "nomanifest");
    }

    private void testLegacyPredict(Channel channel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/noop/predict?data=test");
        channel.writeAndFlush(req);

        latch.await();
        Assert.assertEquals(result, "OK");
    }

    private void testInvalidRootRequest() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    private void testInvalidInferenceUri() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
    }

    private void testInvalidDescribeModel() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/InvalidModel");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    private void testInvalidPredictionsUri() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
    }

    private void testPredictionsModelNotFound() throws InterruptedException {
        Channel channel = connect(false);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions/InvalidModel");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    private void testInvalidManagementUri() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
    }

    private void testInvalidModelsMethod() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    private void testInvalidModelMethod() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
        Assert.assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
    }

    private void testDescribeModelNotFound() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/InvalidModel");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: InvalidModel");
    }

    private void testRegisterModelMissingUrl() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(resp.getMessage(), "Parameter url is required.");
    }

    private void testRegisterModelInvalidRuntime() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=InvalidUrl&runtime=InvalidRuntime");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(resp.getMessage(), "Invalid RuntimeType value: InvalidRuntime");
    }

    private void testRegisterModelNotFound() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models?url=InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found in model store: InvalidUrl");
    }

    private void testRegisterModelConflict() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop.mar&model_name=noop_v1.0&runtime=python&synchronous=false");
        channel.writeAndFlush(req);
        latch.await();

        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=noop.mar&model_name=noop_v1.0&runtime=python&synchronous=false");
        channel.writeAndFlush(req);
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.CONFLICT.code());
        Assert.assertEquals(
                resp.getMessage(), "Model version 1.11 is already registered for model noop_v1.0");
    }

    private void testRegisterModelMalformedUrl() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=http%3A%2F%2Flocalhost%3Aaaaa");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Invalid model url: http://localhost:aaaa");
    }

    private void testRegisterModelConnectionFailed() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=http%3A%2F%2Flocalhost%3A18888%2Ffake.mar&synchronous=false");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(),
                "Failed to download model from: http://localhost:18888/fake.mar");
    }

    private void testRegisterModelHttpError() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=https%3A%2F%2Flocalhost%3A8443%2Ffake.mar&synchronous=false");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(),
                "Failed to download model from: https://localhost:8443/fake.mar, code: 404");
    }

    private void testRegisterModelInvalidPath() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=..%2Ffake.mar&synchronous=false");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Relative path is not allowed in url: ../fake.mar");
    }

    private void testScaleModelNotFound() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models/fake");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: fake");
    }

    private void testUnregisterModelNotFound() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/fake");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Model not found: fake");
    }

    private void testUnregisterModelTimeout()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        Channel channel = connect(true);
        setConfiguration("unregister_model_timeout", "0");

        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/noop_v1.0");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.REQUEST_TIMEOUT.code());
        Assert.assertEquals(resp.getMessage(), "Timed out while cleaning resources: noop_v1.0");

        channel = connect(true);
        setConfiguration("unregister_model_timeout", "120");

        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/noop_v1.0");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
    }

    private void testScaleModelFailure() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        httpStatus = null;
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=init-error.mar&model_name=init-error&synchronous=false");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);

        httpStatus = null;
        result = null;
        latch = new CountDownLatch(1);
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/init-error?synchronous=true&min_worker=1");
        channel.writeAndFlush(req);
        latch.await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);

        Assert.assertEquals(httpStatus, HttpResponseStatus.INTERNAL_SERVER_ERROR);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.INTERNAL_SERVER_ERROR.code());
        Assert.assertEquals(resp.getMessage(), "Failed to start workers");
    }

    private void testLoadingMemoryError() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);
        result = null;
        latch = new CountDownLatch(1);
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=loading-memory-error.mar&model_name=memory_error&runtime=python&initial_workers=1&synchronous=true");
        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.INSUFFICIENT_STORAGE);
        channel.close();
    }

    private void testPredictionMemoryError() throws InterruptedException {
        // Load the model
        Channel channel = connect(true);
        Assert.assertNotNull(channel);
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=prediction-memory-error.mar&model_name=pred-err&runtime=python&initial_workers=1&synchronous=true");
        channel.writeAndFlush(req);
        latch.await();
        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
        channel.close();

        // Test for prediction
        channel = connect(false);
        Assert.assertNotNull(channel);
        result = null;
        latch = new CountDownLatch(1);
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/pred-err");
        req.content().writeCharSequence("data=invalid_output", CharsetUtil.UTF_8);

        channel.writeAndFlush(req);
        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.INSUFFICIENT_STORAGE);
        channel.close();

        // Unload the model
        channel = connect(true);
        httpStatus = null;
        latch = new CountDownLatch(1);
        Assert.assertNotNull(channel);
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/pred-err");
        channel.writeAndFlush(req);
        latch.await();
        Assert.assertEquals(httpStatus, HttpResponseStatus.OK);
    }

    private void testErrorBatch() throws InterruptedException {
        Channel channel = connect(true);
        Assert.assertNotNull(channel);

        httpStatus = null;
        result = null;
        latch = new CountDownLatch(1);
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=error_batch.mar&model_name=err_batch&initial_workers=1&synchronous=true");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse status = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(status.getStatus(), "Workers scaled");

        channel.close();

        channel = connect(false);
        Assert.assertNotNull(channel);

        result = null;
        latch = new CountDownLatch(1);
        httpStatus = null;
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/err_batch");
        req.content().writeCharSequence("data=invalid_output", CharsetUtil.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers()
                .set(
                        HttpHeaderNames.CONTENT_TYPE,
                        HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED);
        channel.writeAndFlush(req);

        latch.await();

        Assert.assertEquals(httpStatus, HttpResponseStatus.INSUFFICIENT_STORAGE);
        Assert.assertEquals(result, "Invalid response");
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

    private void testCheckpoint(Channel managementChannel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/checkpoints/checkpoint1";
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        managementChannel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Checkpoint checkpoint1 saved succsesfully");
    }

    private void testGetCheckpoint(Channel managementChannel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/checkpoints/checkpoint1";
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, requestURL);
        managementChannel.writeAndFlush(req);
        latch.await();

        JsonParser parser = new JsonParser();
        JsonObject json = (JsonObject) parser.parse(result);
        json.remove("created");
        Assert.assertEquals(json.toString(), getCheckpointAPIResult);
    }

    private void testGetAllCheckpoint(Channel managementChannel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/checkpoints";
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, requestURL);
        managementChannel.writeAndFlush(req);
        latch.await();

        JsonParser parser = new JsonParser();
        JsonArray json = (JsonArray) parser.parse(result);
        for (int i = 0; i < json.size(); i++) {
            JsonObject jo = json.get(i).getAsJsonObject();
            jo.remove("created");
        }

        Assert.assertEquals(json.toString(), getAllCheckpointAPIResult);
    }

    private void testRestoreCheckpoint(Channel managementChannel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/checkpoints/checkpoint1/restart";
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        managementChannel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Checkpoint checkpoint1 started succsesfully");
    }

    private void testRemoveCheckpoint(Channel managementChannel) throws InterruptedException {
        result = null;
        latch = new CountDownLatch(1);
        String requestURL = "/checkpoints/checkpoint1";
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, requestURL);
        managementChannel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Checkpoint checkpoint1 removed succsesfully");
    }

    private Channel connect(boolean management) {
        Logger logger = LoggerFactory.getLogger(ModelServerTest.class);

        final Connector connector = configManager.getListener(management);
        try {
            Bootstrap b = new Bootstrap();
            final SslContext sslCtx =
                    SslContextBuilder.forClient()
                            .trustManager(InsecureTrustManagerFactory.INSTANCE)
                            .build();
            b.group(Connector.newEventLoopGroup(1))
                    .channel(connector.getClientChannel())
                    .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 10000)
                    .handler(
                            new ChannelInitializer<Channel>() {
                                @Override
                                public void initChannel(Channel ch) {
                                    ChannelPipeline p = ch.pipeline();
                                    if (connector.isSsl()) {
                                        p.addLast(sslCtx.newHandler(ch.alloc()));
                                    }
                                    p.addLast(new ReadTimeoutHandler(30));
                                    p.addLast(new HttpClientCodec());
                                    p.addLast(new HttpContentDecompressor());
                                    p.addLast(new ChunkedWriteHandler());
                                    p.addLast(new HttpObjectAggregator(6553600));
                                    p.addLast(new TestHandler());
                                }
                            });

            return b.connect(connector.getSocketAddress()).sync().channel();
        } catch (Throwable t) {
            logger.warn("Connect error.", t);
        }
        return null;
    }

    @ChannelHandler.Sharable
    private class TestHandler extends SimpleChannelInboundHandler<FullHttpResponse> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, FullHttpResponse msg) {
            httpStatus = msg.status();
            result = msg.content().toString(StandardCharsets.UTF_8);
            headers = msg.headers();
            latch.countDown();
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            Logger logger = LoggerFactory.getLogger(TestHandler.class);
            logger.error("Unknown exception", cause);
            ctx.close();
            latch.countDown();
        }
    }
}
