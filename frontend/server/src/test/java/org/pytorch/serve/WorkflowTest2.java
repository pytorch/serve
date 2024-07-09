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
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
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

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSyntaxException;

public class WorkflowTest2 {

     static {
        TestUtils.init();
    }

    private ConfigManager configManager;
    private ModelServer server;
    private static final String CURR_DIR = System.getProperty("user.dir");

    @BeforeClass
    public void beforeClass()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        System.setProperty("tsConfigFile", "src/test/resources/config_workflow.properties");
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

    private void testLoadWorkflow(String url, String workflowName) throws InterruptedException {
        Channel channel = TestUtils.getManagementChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        String managementKey = readManagementKey();
        TestUtils.registerWorkflowToken(channel, url, workflowName, false, managementKey);
        TestUtils.getLatch().await();

        StatusResponse resp = JsonUtils.GSON.fromJson(TestUtils.getResult(), StatusResponse.class);
        Assert.assertEquals(
                resp.getStatus(),
                "Workflow " + workflowName + " has been registered and scaled successfully.");
    }

    public static String readManagementKey() {
        String jsonFilePath = Paths.get(CURR_DIR, "key_file.json").toString();
        try (FileReader reader = new FileReader(jsonFilePath)) {
            JsonElement jsonElement = JsonParser.parseReader(reader);
            if (jsonElement.isJsonObject()) {
                JsonObject jsonData = jsonElement.getAsJsonObject();
                return getJsonValue(jsonData, "management", "key");
            }
        } catch (IOException | JsonSyntaxException e) {
            e.printStackTrace();
        }
        return "Error reading file";
    }

    private static String getJsonValue(JsonObject jsonObject, String primaryKey, String secondaryKey) {
        JsonObject primaryObject = jsonObject.has(primaryKey) ? jsonObject.getAsJsonObject(primaryKey) : null;
        return primaryObject != null && primaryObject.has(secondaryKey) ? primaryObject.get(secondaryKey).getAsString() : "NOT_PRESENT";
    }


}
