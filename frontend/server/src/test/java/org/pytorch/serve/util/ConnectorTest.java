package org.pytorch.serve.util;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ConnectorTest {

    @Test
    public void testValidManagementAddress() {
        Connector conn = Connector.parse("http://127.0.0.1:45245", true);

        Assert.assertEquals(conn.getSocketType(), "tcp");
        Assert.assertEquals(conn.getSocketPath(), "45245");
        Assert.assertTrue(conn.isManagement());
        Assert.assertFalse(conn.isSsl());
        Assert.assertFalse(conn.isUds());
    }

    @Test
    public void testValidInferenceAddress() {
        Connector conn = Connector.parse("http://127.0.0.1:45245", false);

        Assert.assertEquals(conn.getSocketType(), "tcp");
        Assert.assertEquals(conn.getSocketPath(), "45245");
        Assert.assertFalse(conn.isManagement());
        Assert.assertFalse(conn.isSsl());
        Assert.assertFalse(conn.isUds());
    }

    @Test
    public void testSSLInferenceAddress() {
        Connector conn = Connector.parse("https://127.0.0.1:45245", false);

        Assert.assertEquals(conn.getSocketType(), "tcp");
        Assert.assertEquals(conn.getSocketPath(), "45245");
        Assert.assertFalse(conn.isManagement());
        Assert.assertTrue(conn.isSsl());
        Assert.assertFalse(conn.isUds());
    }

    @Test
    public void testInvalidPort() {
        try {
            Connector conn = Connector.parse("https://127.0.0.1:66666", false);
        } catch (Exception e) {
            Assert.assertEquals(e.getClass(), IllegalArgumentException.class);
            Assert.assertEquals(e.getMessage(), "Invalid port number: https://127.0.0.1:66666");
        }
    }

    @Test
    public void testInvalidBindindAddress() {
        try {
            Connector conn = Connector.parse("https://Incorrect?*binding:11111", false);
        } catch (Exception e) {
            Assert.assertEquals(e.getClass(), IllegalArgumentException.class);
            Assert.assertEquals(e.getMessage(), "Invalid binding address");
        }
    }

    @Test
    public void testUDSAddress() {
        Connector conn = Connector.parse("unix:/tmp/management.sock", false);
        Assert.assertEquals(conn.getSocketType(), "unix");
        Assert.assertEquals(conn.getSocketPath(), "/tmp/management.sock");
        Assert.assertFalse(conn.isManagement());
        Assert.assertFalse(conn.isSsl());
        Assert.assertTrue(conn.isUds());
    }

    @Test
    public void testDefaultManagementAddress() {
        Connector conn = Connector.parse("http://127.0.0.1", true);
        Assert.assertEquals(conn.getSocketType(), "tcp");
        Assert.assertEquals(conn.getSocketPath(), "8081");
        Assert.assertTrue(conn.isManagement());
        Assert.assertFalse(conn.isSsl());
        Assert.assertFalse(conn.isUds());
    }

    @Test
    public void testDefaultInferenceAddress() {
        Connector conn = Connector.parse("http://127.0.0.1", false);
        Assert.assertEquals(conn.getSocketType(), "tcp");
        Assert.assertEquals(conn.getSocketPath(), "80");
        Assert.assertFalse(conn.isManagement());
        Assert.assertFalse(conn.isSsl());
        Assert.assertFalse(conn.isUds());
    }

    @Test
    public void testDefaultSSLManagementAddress() {
        Connector conn = Connector.parse("https://127.0.0.1", true);
        Assert.assertEquals(conn.getSocketType(), "tcp");
        Assert.assertEquals(conn.getSocketPath(), "8444");
        Assert.assertTrue(conn.isManagement());
        Assert.assertTrue(conn.isSsl());
        Assert.assertFalse(conn.isUds());
    }

    @Test
    public void testDefaultSSLInferenceAddress() {
        Connector conn = Connector.parse("https://127.0.0.1", false);
        Assert.assertEquals(conn.getSocketType(), "tcp");
        Assert.assertEquals(conn.getSocketPath(), "443");
        Assert.assertFalse(conn.isManagement());
        Assert.assertTrue(conn.isSsl());
        Assert.assertFalse(conn.isUds());
    }
}
