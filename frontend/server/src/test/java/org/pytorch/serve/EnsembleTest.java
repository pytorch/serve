package org.pytorch.serve;

import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import org.apache.commons.io.IOUtils;
import org.pytorch.serve.ensemble.Dag;
import org.pytorch.serve.ensemble.Node;
import org.pytorch.serve.ensemble.WorkflowModel;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.snapshot.InvalidSnapshotException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.JsonUtils;
import org.pytorch.serve.workflow.WorkflowManager;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

public class EnsembleTest {

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

    @Test(alwaysRun = true)
    public void testDAG() {
        Dag dag = new Dag();

        Node a = new Node("a", new WorkflowModel("a", "url", 1, 1, 10, 50, 1000, 5, null));
        Node b = new Node("b", new WorkflowModel("b", "url", 1, 1, 10, 50, 1000, 5, null));
        Node c = new Node("c", new WorkflowModel("c", "url", 1, 1, 10, 50, 1000, 5, null));
        Node d = new Node("d", new WorkflowModel("d", "url", 1, 1, 10, 50, 1000, 5, null));
        Node e = new Node("e", new WorkflowModel("e", "url", 1, 1, 10, 50, 1000, 5, null));
        Node f = new Node("f", new WorkflowModel("f", "url", 1, 1, 10, 50, 1000, 5, null));

        dag.addNode(a);
        dag.addNode(b);
        dag.addNode(c);
        dag.addNode(d);
        dag.addNode(e);
        dag.addNode(f);

        try {
            dag.addEdge(a, b);
            dag.addEdge(a, c);
            dag.addEdge(a, d);
            dag.addEdge(b, e);
            dag.addEdge(e, f);
            dag.addEdge(c, f);
            dag.addEdge(d, f);
            String[] list = {"a", "b", "c", "d", "e", "f"};
            Assert.assertEquals(dag.validate().toArray(), list);
        } catch (Exception exp) {
            Assert.assertTrue(false);
        }
    }

    @Test(alwaysRun = true)
    public void testInvalidDAG() {
        Dag dag = new Dag();

        Node a = new Node("a", new WorkflowModel("a", "url", 1, 1, 10, 50, 1000, 5, null));
        Node b = new Node("b", new WorkflowModel("b", "url", 1, 1, 10, 50, 1000, 5, null));
        Node c = new Node("c", new WorkflowModel("c", "url", 1, 1, 10, 50, 1000, 5, null));
        Node d = new Node("d", new WorkflowModel("d", "url", 1, 1, 10, 50, 1000, 5, null));
        Node e = new Node("e", new WorkflowModel("e", "url", 1, 1, 10, 50, 1000, 5, null));
        Node f = new Node("f", new WorkflowModel("f", "url", 1, 1, 10, 50, 1000, 5, null));

        dag.addNode(a);
        dag.addNode(b);
        dag.addNode(c);
        dag.addNode(d);
        dag.addNode(e);
        dag.addNode(f);

        try {
            dag.addEdge(a, b);
            dag.addEdge(a, c);
            dag.addEdge(a, d);
            dag.addEdge(b, e);
            dag.addEdge(e, f);
            dag.addEdge(c, f);
            dag.addEdge(d, f);
            dag.addEdge(f, b);
            System.out.println(dag.validate());
            Assert.assertTrue(false);
        } catch (Exception exp) {
            Assert.assertTrue(true);
        }
    }

    @Test(alwaysRun = true)
    public void testWorkflowYaml() throws Exception {
        //         torch-workflow-archiver  --workflow-name test  --spec-file
        //         /Users/demo/git/serve/frontend/server/src/test/resources/workflow_spec.yaml
        // --handler
        //         /Users/demo/git/serve/frontend/server/src/test/resources/workflow_handler.py
        //         --export-path /Users/demo/git/serve/frontend/server/src/test/resources/ -f

        try {
            StatusResponse status =
                    WorkflowManager.getInstance()
                            .registerWorkflow(
                                    "test.war",
                                    "file:///Users/demo/git/serve/frontend/server/src/test/resources/test.war",
                                    300,
                                    300,
                                    true);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            e.printStackTrace();
        }
    }
}
