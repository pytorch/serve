package org.pytorch.serve.archive.utils;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ArchiveUtilsTest {
    @Test
    public void testGetFilenameFromUrlWithFilename() {
        String testFilename = "resnet-18.mar";
        String expectedFilename = "resnet-18.mar";
        Assert.assertEquals(ArchiveUtils.getFilenameFromUrl(testFilename), expectedFilename);
    }

    @Test
    public void testGetFilenameFromUrlWithFilepath() {
        String testFilepath = "/home/ubuntu/model_store/resnet-18.mar";
        String expectedFilename = "resnet-18.mar";
        Assert.assertEquals(ArchiveUtils.getFilenameFromUrl(testFilepath), expectedFilename);
    }

    @Test
    public void testGetFilenameFromUrlWithUrl() {
        String testFileUrl = "https://torchserve.pytorch.org/mar_files/resnet-18.mar";
        String expectedFilename = "resnet-18.mar";
        Assert.assertEquals(ArchiveUtils.getFilenameFromUrl(testFileUrl), expectedFilename);
    }

    @Test
    public void testGetFilenameFromUrlWithS3PresignedUrl() {
        String testFileUrl =
                "https://test-account.s3.us-west-2.amazonaws.com/mar_files/resnet-18.mar?"
                        + "response-content-disposition=inline&X-Amz-Security-Token=%2Ftoken%2F"
                        + "&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230614T182131Z&X-Amz-SignedHeaders=host"
                        + "&X-Amz-Expires=43200&X-Amz-Credential=%2Fcredential%2F"
                        + "&X-Amz-Signature=%2Fsignature%2F";
        String expectedFilename = "resnet-18.mar";
        Assert.assertEquals(ArchiveUtils.getFilenameFromUrl(testFileUrl), expectedFilename);
    }

    @Test
    public void testGetFilenameFromUrlWithInvalidUrl() {
        String testFileUrl = "resnet-18.mar/";
        String expectedFilename = "";
        Assert.assertEquals(ArchiveUtils.getFilenameFromUrl(testFileUrl), expectedFilename);
    }
}
