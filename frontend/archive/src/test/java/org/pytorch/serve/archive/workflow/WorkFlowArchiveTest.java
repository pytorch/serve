package org.pytorch.serve.archive.workflow;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class WorkFlowArchiveTest {

    private static final List<String> ALLOWED_URLS_LIST =
            Collections.singletonList("file://.*|http(s)?://.*");

    @BeforeTest
    public void beforeTest() {
        File tmp = FileUtils.getTempDirectory();
        FileUtils.deleteQuietly(new File(tmp, "workflows"));
    }

    @Test
    public void test() throws IOException, DownloadArchiveException, WorkflowException {
        String workflowStore = "src/test/resources/workflows";
        WorkflowArchive archive =
                WorkflowArchive.downloadWorkflow(ALLOWED_URLS_LIST, workflowStore, "smtest.war");
        archive.validate();
        archive.clean();
        Assert.assertEquals(archive.getWorkflowName(), "smtest");
    }

    @Test(
            expectedExceptions = DownloadArchiveException.class,
            expectedExceptionsMessageRegExp =
                    "Failed to download archive from: https://s3\\.amazonaws\\.com/squeezenet_v1\\.1\\.mod")
    public void testAllowedURL() throws WorkflowException, IOException, DownloadArchiveException {
        // test allowed url, return failed to download as file does not exist
        String workflowStore = "src/test/resources/workflows";
        WorkflowArchive.downloadWorkflow(
                ALLOWED_URLS_LIST, workflowStore, "https://s3.amazonaws.com/squeezenet_v1.1.mod");
    }

    @Test(
            expectedExceptions = DownloadArchiveException.class,
            expectedExceptionsMessageRegExp =
                    "Failed to download archive from: https://torchserve\\.pytorch\\.org/mar_files/mnist_non_exist\\.war")
    public void testAllowedMultiUrls()
            throws WorkflowException, IOException, DownloadArchiveException {
        // test multiple urls added to allowed list
        String workflowStore = "src/test/resources/workflows";
        final List<String> customUrlPatternList =
                Arrays.asList(
                        "http(s)?://s3.amazonaws.com.*",
                        "https://torchserve.pytorch.org/mar_files/.*");
        WorkflowArchive.downloadWorkflow(
                customUrlPatternList,
                workflowStore,
                "https://torchserve.pytorch.org/mar_files/mnist_non_exist.war");
    }

    @Test(
            expectedExceptions = WorkflowNotFoundException.class,
            expectedExceptionsMessageRegExp =
                    "Given URL https://torchserve\\.pytorch.org/mar_files/mnist\\.war does not match any allowed URL\\(s\\)")
    public void testBlockedUrl() throws WorkflowException, IOException, DownloadArchiveException {
        // test blocked url
        String workflowStore = "src/test/resources/workflows";
        final List<String> customUrlPatternList =
                Collections.singletonList("http(s)?://s3.amazonaws.com.*");
        WorkflowArchive.downloadWorkflow(
                customUrlPatternList,
                workflowStore,
                "https://torchserve.pytorch.org/mar_files/mnist.war");
    }
}
