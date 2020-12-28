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
import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.snapshot.InvalidSnapshotException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.workflow.messages.DescribeWorkflowResponse;
import org.pytorch.serve.workflow.messages.ListWorkflowResponse;
import org.testng.Assert;
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
    public void testWorkflowPrediction() throws InterruptedException {
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
}
