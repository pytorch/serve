package org.pytorch.serve.plugins.test.prometheus;

import org.pytorch.serve.plugins.endpoint.prometheus.PrometheusMetricEventListener;
import org.pytorch.serve.plugins.endpoint.prometheus.PrometheusMetricManager;
import org.pytorch.serve.plugins.endpoint.prometheus.PrometheusMetrics;
import org.pytorch.serve.servingsdk.metrics.*;
import org.testng.annotations.Test;
import org.testng.annotations.BeforeSuite;
import org.testng.Assert;
import org.mockito.Mockito;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.http.Response;
import org.pytorch.serve.servingsdk.Context;
import org.pytorch.serve.servingsdk.Model;
import org.pytorch.serve.servingsdk.Worker;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;

/**
 * Unit test for simple App.
 */
public class PluginTest {
    Context c;
    Model m;
    Worker w;
    PrometheusMetrics pm;
    BaseMetric bm;
    BaseDimension bd1, bd2;
    MetricLogEvent mle;
    MetricEventListenerRegistry melr;
    MetricEventPublisher mep;
    MetricEventListener mel;
    Request req;
    Response rsp;
    ByteArrayOutputStream outputStream;
    Endpoint ea;

    @BeforeSuite
    public void beforeSuite() throws IOException {
        c = Mockito.mock(Context.class);
        m = Mockito.mock(Model.class);
        w = Mockito.mock(Worker.class);
        ea = Mockito.mock(Endpoint.class);
        pm = Mockito.mock(PrometheusMetrics.class);
        bm = Mockito.mock(BaseMetric.class);
        bd1 = Mockito.mock(BaseDimension.class);
        bd2 = Mockito.mock(BaseDimension.class);
        mle = Mockito.mock(MetricLogEvent.class);
        melr = Mockito.mock(MetricEventListenerRegistry.class);
        mep = Mockito.mock(MetricEventPublisher.class);
        mel = Mockito.mock(MetricEventListener.class);
        req = Mockito.mock(Request.class);
        rsp = Mockito.mock(Response.class);
        outputStream = new ByteArrayOutputStream();

        Properties p = new Properties();
        p.setProperty("Hello", "World");
        c.getConfig();
        Mockito.when(c.getConfig()).thenReturn(p);
        Mockito.when(rsp.getOutputStream()).thenReturn(outputStream);
    }


    @Test
    private void testEndpoint() throws IOException {
        PrometheusMetrics b = new PrometheusMetrics();
        PrometheusMetricManager a = PrometheusMetricManager.getInstance();
        a.incInferCount(1, "noop", "1.0");
        a.incBackendResponseLatency(3, "noop", "1.0");
        b.doGet(req, rsp, c);
        Assert.assertTrue(outputStream.toString().matches("(?is).*ts_backend_reponse_latency_milliseconds_count.*model_name=\"noop\",model_version=\"1.0\",} 1.0.*"));
        Assert.assertTrue(outputStream.toString().matches("(?is).*ts_backend_reponse_latency_milliseconds_sum.*model_name=\"noop\",model_version=\"1.0\",} 3.0.*"));
        Assert.assertTrue(outputStream.toString().matches("(?is).*ts_inference_requests_total.*model_name=\"noop\",model_version=\"1.0\",} 1.0.*"));
        outputStream.reset();

        Mockito.when(mle.getMetric()).thenReturn(bm);
        Mockito.when(bm.getMetricName()).thenReturn(InbuiltMetricsRegistry.INFERENCE);
        Mockito.when(bm.getValue()).thenReturn("1");
        Mockito.when(bm.getDimensions()).thenReturn(Arrays.asList(new BaseDimension[]{bd1, bd2}));
        Mockito.when(bd1.getName()).thenReturn(DimensionRegistry.MODELNAME);
        Mockito.when(bd1.getValue()).thenReturn("noop");
        Mockito.when(bd2.getName()).thenReturn(DimensionRegistry.MODELVERSION);
        Mockito.when(bd2.getValue()).thenReturn("1.0");


        PrometheusMetricEventListener pel = new PrometheusMetricEventListener();
        pel.handle(mle);
        b.doGet(req, rsp, c);
        Assert.assertTrue(outputStream.toString().matches("(?is).*ts_inference_requests_total.*model_name=\"noop\",model_version=\"1.0\",} 2.0.*"));
        outputStream.reset();

    }

}
