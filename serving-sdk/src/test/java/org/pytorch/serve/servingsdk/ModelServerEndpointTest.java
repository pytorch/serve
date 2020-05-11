

package org.pytorch.serve.servingsdk;


import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.mockito.stubbing.Answer;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;
import org.pytorch.serve.servingsdk.http.Response;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

/**
 * Unit test for simple App.
 */
public class ModelServerEndpointTest {
    Context c;
    Model m;
    Worker w;
    ModelServerEndpoint mse;
    Request req;
    Response rsp;
    ByteArrayOutputStream outputStream;
    Endpoint ea;

    @Before
    public void beforeSuite() throws IOException {
        c = Mockito.mock(Context.class);
        m = Mockito.mock(Model.class);
        w = Mockito.mock(Worker.class);
        ea = Mockito.mock(Endpoint.class);
        mse = Mockito.mock(ModelServerEndpoint.class);
        req = Mockito.mock(Request.class);
        rsp = Mockito.mock(Response.class);
        outputStream = new ByteArrayOutputStream();

        Properties p = new Properties();
        HashMap<String, Model> map = new HashMap<>();
        List<Worker> l = new ArrayList<>();
        map.put("squeezenet", m);
        p.setProperty("Hello", "World");
        c.getConfig();
        l.add(w);

        Mockito.when(c.getConfig()).thenReturn(p);
        Mockito.when(c.getModels()).thenReturn(map);
        Mockito.when(m.getModelWorkers()).thenReturn(l);
        Mockito.when(m.getModelHandler()).thenReturn("mxnet_service:handle");
        Mockito.when(m.getModelUrl()).thenReturn("/tmp/model/squeezenet.mar");
        Mockito.when(m.getModelName()).thenReturn("squeezenet");
        Mockito.when(w.getWorkerMemory()).thenReturn((long)100);
        Mockito.when(w.isRunning()).thenReturn(false);
        Mockito.when(ea.urlPattern()).thenReturn("myEndpoint");
        Mockito.when(ea.description()).thenReturn("This is a test endpoint");
        Mockito.when(ea.endpointType()).thenReturn(EndpointTypes.INFERENCE);
        Mockito.when(rsp.getOutputStream()).thenReturn(outputStream);
    }

    @Test
    public void test() throws IOException {
        testContextInterface();
        testEndpointAnnotation();
        testEndpointInterface();
    }

    private void testEndpointInterface() throws IOException {
        Class ep = ModelServerEndpoint.class;
        Assert.assertEquals(4, ep.getDeclaredMethods().length);
        for(Method m : ep.getDeclaredMethods()) {
            switch (m.getName()) {
                case "doGet":
                case "doPost":
                case "doDelete":
                case "doPut":
                    break;
                default:
                    Assert.fail("Invalid method found");
            }
        }

        // Check signatures
        Mockito.doAnswer((Answer) i -> {
            Object rq = i.getArguments()[0];
            Object rs = i.getArguments()[1];
            Object ctx = i.getArguments()[2];

            ((Response)rs).getOutputStream().write("This is a test".getBytes());
            return null;
        }).when(mse).doGet(req, rsp, c);

        mse.doGet(req, rsp, c);

        Assert.assertEquals("This is a test", outputStream.toString());
        outputStream.reset();

        // Check signatures
        Mockito.doAnswer((Answer) i -> {
            Object rq = i.getArguments()[0];
            Object rs = i.getArguments()[1];
            Object ctx = i.getArguments()[2];
            ((Response)rs).getOutputStream().write("This is a test".getBytes());
            return null;
        }).when(mse).doPost(req, rsp, c);

        mse.doPost(req, rsp, c);

        Assert.assertEquals("This is a test", outputStream.toString());
        outputStream.reset();

        // Check signatures
        Mockito.doAnswer((Answer) i -> {
            Object rq = i.getArguments()[0];
            Object rs = i.getArguments()[1];
            Object ctx = i.getArguments()[2];

            ((Response)rs).getOutputStream().write("This is a test".getBytes());
            return null;
        }).when(mse).doPut(req, rsp, c);

        mse.doPut(req, rsp, c);

        Assert.assertEquals("This is a test", outputStream.toString());
        outputStream.reset();

        // Check signatures
        Mockito.doAnswer((Answer) i -> {
            Object rq = i.getArguments()[0];
            Object rs = i.getArguments()[1];
            Object ctx = i.getArguments()[2];

            ((Response)rs).getOutputStream().write("This is a test".getBytes());
            return null;
        }).when(mse).doDelete(req, rsp, c);

        mse.doDelete(req, rsp, c);

        Assert.assertEquals("This is a test", outputStream.toString());
    }

    private void testEndpointAnnotation() {
        Assert.assertEquals(3, Endpoint.class.getDeclaredMethods().length);
        Assert.assertEquals("myEndpoint", ea.urlPattern());
        Assert.assertEquals(EndpointTypes.INFERENCE, ea.endpointType());
        Assert.assertEquals("This is a test endpoint", ea.description());
        Assert.assertEquals(3, EndpointTypes.class.getFields().length);
    }

    private void testWorkerInterface(Worker w) {
        Assert.assertNotNull(w);
        Assert.assertFalse( w.isRunning());
        Assert.assertEquals(100, w.getWorkerMemory());
        Assert.assertEquals(2, Worker.class.getDeclaredMethods().length);
    }

    private void testModelInterface(Model m) {
        Assert.assertEquals("squeezenet", m.getModelName());
        Assert.assertEquals("/tmp/model/squeezenet.mar", m.getModelUrl());
        Assert.assertEquals("mxnet_service:handle", m.getModelHandler());
        Assert.assertEquals(1, m.getModelWorkers().size());
        Assert.assertEquals(4, Model.class.getDeclaredMethods().length);
        testWorkerInterface(m.getModelWorkers().get(0));
    }

    private void testContextInterface() {
        Assert.assertNotNull(c.getModels());
        Assert.assertTrue(c.getModels().containsKey("squeezenet"));
        Assert.assertTrue(c.getConfig().containsKey("Hello"));
        Assert.assertEquals("World", c.getConfig().getProperty("Hello"));
        Assert.assertEquals(2, Context.class.getDeclaredMethods().length);
        testModelInterface(c.getModels().get("squeezenet"));
    }
}
