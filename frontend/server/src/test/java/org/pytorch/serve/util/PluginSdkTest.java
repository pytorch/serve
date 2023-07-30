package org.pytorch.serve.util;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.handler.codec.http.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.impl.ModelServerRequest;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PluginSdkTest {
    @Test
    public void testReadRequestBodyFromPlugin() throws IOException {
        ByteBuf buf = Unpooled.directBuffer();
        buf.writeBytes("test".getBytes());
        FullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "", buf);
        Request request = new ModelServerRequest(req, new QueryStringDecoder(""));
        String line =
                new BufferedReader(new InputStreamReader(request.getInputStream())).readLine();
        Assert.assertEquals(line, "test");
    }
}
