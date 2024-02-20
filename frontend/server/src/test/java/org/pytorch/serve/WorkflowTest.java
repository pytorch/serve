package org.pytorch.serve;

import io.netty.channel.Channel;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.util.CharsetUtil;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.File;
import java.io.IOException;
import java.security.GeneralSecurityException;
import java.util.concurrent.CountDownLatch;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.http.ErrorResponse;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.metrics.MetricCache;
import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.snapshot.InvalidSnapshotException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.workflow.messages.DescribeWorkflowResponse;
import org.pytorch.serve.workflow.messages.ListWorkflowResponse;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class WorkflowTest {

    static {
        TestUtils.init();
    }

    private ConfigManager configManager;
    private ModelServer server;

    @BeforeClass
    public void beforeClass()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        System.setProperty("tsConfigFile", "src/test/resources/config.properties");
        FileUtils.cleanDirectory(new File(System.getProperty("LOG_LOCATION"), "config"));

        ConfigManager.init(new ConfigManager.Arguments());
        configManager = ConfigManager.getInstance();
        PluginsManager.getInstance().initialize();
        MetricCache.init();

        InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);
        configManager.setInitialWorkerPort(10000);
        configManager.setProperty("load_models", "");
        server = new ModelServer(configManager);
        server.startRESTserver();
    }

    @AfterClass
    public void afterClass() throws InterruptedException {
        TestUtils.closeChannels();
        server.stop();
    }

    @Test
    public void testRegisterWorkflow() throws InterruptedException {
        testLoadWorkflow("smtest.war", "smtest");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterWorkflow"})
    public void testListWorkflow() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.listWorkflow(channel);
        TestUtils.getLatch().await();

        ListWorkflowResponse resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), ListWorkflowResponse.class);
        Assert.assertEquals(resp.getWorkflows().size(), 1);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testListWorkflow"})
    public void testDescribeWorkflow() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.describeWorkflow(channel, "smtest");
        TestUtils.getLatch().await();

        DescribeWorkflowResponse[] resp =
                JsonUtils.GSON.fromJson(TestUtils.getResult(), DescribeWorkflowResponse[].class);
        Assert.assertEquals(resp.length, 1);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeWorkflow"})
    public void testWorkflowPrediction() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String requestURL = "/wfpredict/" + "smtest";
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
        Assert.assertEquals(TestUtils.getResult(), "0");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testWorkflowPrediction"})
    public void testUnregisterWorkflow() throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterWorkflow(channel, "smtest", false);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Workflow \"smtest\" unregistered");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterWorkflow"})
    public void testLoadWorkflowFromFileURI() throws InterruptedException, IOException {
        String curDir = System.getProperty("user.dir");
        File curDirFile = new File(curDir);
        String parent = curDirFile.getParent();

        String source = configManager.getWorkflowStore() + "/smtest.war";
        String destination = parent + "/archive/smtest1.war";
        File sourceFile = new File(source);
        File destinationFile = new File(destination);
        String fileUrl = "";
        FileUtils.copyFile(sourceFile, destinationFile);
        fileUrl = "file:///" + destination;
        testLoadWorkflow(fileUrl, "smtest1");
        Assert.assertTrue(new File(configManager.getWorkflowStore(), "smtest1.war").exists());
        FileUtils.deleteQuietly(destinationFile);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testLoadWorkflowFromFileURI"})
    public void testUnregisterFileURIWorkflow() throws InterruptedException {
        testUnregisterWorkflow("smtest1");
        Assert.assertFalse(new File(configManager.getWorkflowStore(), "smtest1.war").exists());
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterFileURIWorkflow"})
    public void testRegisterWorkflowMissingUrl() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/workflows");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(resp.getMessage(), "Parameter url is required.");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterWorkflowMissingUrl"})
    public void testRegisterWorkflowNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/workflows?url=InvalidUrl");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Workflow not found in workflow store: InvalidUrl");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterWorkflowNotFound"})
    public void testRegisterWorkflowConflict() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        TestUtils.setLatch(new CountDownLatch(1));
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?url=smtest.war&workflow_name=smtest");
        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?url=smtest.war&workflow_name=smtest");
        channel.writeAndFlush(req);
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(resp.getCode(), HttpResponseStatus.CONFLICT.code());
        Assert.assertEquals(resp.getMessage(), "Workflow smtest is already registered.");
        TestUtils.unregisterWorkflow(channel, "smtest", false);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterWorkflowConflict"})
    public void testRegisterWorkflowMalformedUrl() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?url=http%3A%2F%2Flocalhost%3Aaaaa");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(), "Failed to download archive from: http://localhost:aaaa");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterWorkflowMalformedUrl"})
    public void testRegisterWorkflowConnectionFailed() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?url=http%3A%2F%2Flocalhost%3A18888%2Ffake.war&synchronous=false");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(),
                "Failed to download archive from: http://localhost:18888/fake.war");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterWorkflowConnectionFailed"})
    public void testRegisterWorkflowHttpError() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?url=https%3A%2F%2Flocalhost%3A8443%2Ffake.war&synchronous=false");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
        Assert.assertEquals(
                resp.getMessage(),
                "Failed to download archive from: https://localhost:8443/fake.war");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterWorkflowHttpError"})
    public void testRegisterWorkflowInvalidPath() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?url=..%2Ffake.war&synchronous=false");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Relative path is not allowed in url: ../fake.war");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testRegisterWorkflowInvalidPath"})
    public void testUnregisterWorkflowNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        TestUtils.unregisterWorkflow(channel, "fake", true);

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Workflow not found: fake");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testUnregisterWorkflowNotFound"})
    public void testDescribeWorkflowNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        TestUtils.describeWorkflow(channel, "fake");

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Workflow not found: fake");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testDescribeWorkflowNotFound"})
    public void testPredictionWorkflowNotFound() throws InterruptedException {
        Channel channel = TestUtils.connect(ConnectorType.INFERENCE_CONNECTOR, configManager);
        Assert.assertNotNull(channel);

        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/wfpredict/InvalidModel");
        channel.writeAndFlush(req).sync();
        channel.closeFuture().sync();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);

        Assert.assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
        Assert.assertEquals(resp.getMessage(), "Workflow not found: InvalidModel");
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionWorkflowNotFound"})
    public void testWorkflowWithInvalidFileURI() throws InterruptedException, IOException {
        String invalidFileUrl = "file:///InvalidUrl";
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        TestUtils.setHttpStatus(null);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerWorkflow(channel, invalidFileUrl, "invalid_file_url", false);
        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.BAD_REQUEST);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testWorkflowWithInvalidFileURI"})
    public void testWorkflowWithCustomPythonDependencyModel()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        TestUtils.setConfiguration(configManager, "install_py_dep_per_model", "true");
        testLoadWorkflow("custom_python_dep.war", "custom_python_dep");
        TestUtils.setConfiguration(configManager, "install_py_dep_per_model", "false");
        TestUtils.unregisterWorkflow(channel, "custom_python_dep", false);
        channel.close().sync();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testWorkflowWithCustomPythonDependencyModel"})
    public void testWorkflowWithInvalidCustomPythonDependencyModel()
            throws InterruptedException, NoSuchFieldException, IllegalAccessException {
        TestUtils.setConfiguration(configManager, "install_py_dep_per_model", "true");
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerWorkflow(
                channel, "custom_invalid_python_dep.war", "custom_invalid_python_dep", false);
        TestUtils.getLatch().await();

        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.INTERNAL_SERVER_ERROR);
        Assert.assertEquals(
                resp.getMessage(),
                "Workflow custom_invalid_python_dep has failed to register. Failures: [Workflow Node custom_invalid_python_dep__custom_invalid_python_dep failed to register. Details: Custom pip package installation failed for model custom_invalid_python_dep__custom_invalid_python_dep]");
        TestUtils.setConfiguration(configManager, "install_py_dep_per_model", "false");
        channel.close().sync();
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testWorkflowWithInvalidCustomPythonDependencyModel"})
    public void testPredictionMemoryError() throws InterruptedException {
        // Load the model
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerWorkflow(channel, "prediction-memory-error.war", "pred-err", false);
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
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/wfpredict/pred-err");
        req.content().writeCharSequence("data=invalid_output", CharsetUtil.UTF_8);

        channel.writeAndFlush(req);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.INTERNAL_SERVER_ERROR);
        channel.close().sync();

        // Unload the workflow
        channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        TestUtils.setHttpStatus(null);
        TestUtils.setLatch(new CountDownLatch(1));
        Assert.assertNotNull(channel);

        TestUtils.unregisterWorkflow(channel, "pred-err", false);
        TestUtils.getLatch().await();
        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.OK);
    }

    @Test(
            alwaysRun = true,
            dependsOnMethods = {"testPredictionMemoryError"})
    public void testLoadingMemoryError() throws InterruptedException, SkipException {
        if (System.getProperty("os.name").startsWith("Win")) {
            throw new SkipException("Test skipped on Windows");
        }
        Channel channel = TestUtils.connect(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
        Assert.assertNotNull(channel);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));

        TestUtils.registerWorkflow(channel, "loading-memory-error.war", "memory_error", false);
        TestUtils.getLatch().await();

        Assert.assertEquals(TestUtils.getHttpStatus(), HttpResponseStatus.INTERNAL_SERVER_ERROR);
        ErrorResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), ErrorResponse.class);
        Assert.assertEquals(
                resp.getMessage(),
                "Workflow memory_error has failed to register. Failures: [Failed to start workers for model memory_error__loading-memory-error version: 1.0]");
        channel.close().sync();
    }

    private void testLoadWorkflow(String url, String workflowName) throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.registerWorkflow(channel, url, workflowName, false);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(
                resp.getStatus(),
                "Workflow " + workflowName + " has been registered and scaled successfully.");
    }

    private void testUnregisterWorkflow(String workflowName) throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        TestUtils.unregisterWorkflow(channel, workflowName, false);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(resp.getStatus(), "Workflow \"" + workflowName + "\" unregistered");
    }
}
