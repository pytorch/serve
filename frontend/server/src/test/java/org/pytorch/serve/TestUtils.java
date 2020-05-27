package org.pytorch.serve;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpContentDecompressor;
import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import io.netty.handler.stream.ChunkedWriteHandler;
import io.netty.handler.timeout.ReadTimeoutHandler;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.concurrent.CountDownLatch;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class TestUtils {

    static CountDownLatch latch;
    static HttpResponseStatus httpStatus;
    static String result;
    static HttpHeaders headers;
    private static Channel inferenceChannel;
    private static Channel managementChannel;

    private TestUtils() {}

    public static void init() {
        // set up system properties for local IDE debug
        if (System.getProperty("tsConfigFile") == null) {
            System.setProperty("tsConfigFile", "src/test/resources/config.properties");
        }
        if (System.getProperty("METRICS_LOCATION") == null) {
            System.setProperty("METRICS_LOCATION", "build/logs");
        }
        if (System.getProperty("LOG_LOCATION") == null) {
            System.setProperty("LOG_LOCATION", "build/logs");
        }

        try {
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(null, InsecureTrustManagerFactory.INSTANCE.getTrustManagers(), null);

            HttpsURLConnection.setDefaultSSLSocketFactory(context.getSocketFactory());

            HttpsURLConnection.setDefaultHostnameVerifier((s, sslSession) -> true);
        } catch (GeneralSecurityException e) {
            // ignore
        }
    }

    public static HttpHeaders getHeaders() {
        return headers;
    }

    public static void setHeaders(HttpHeaders newHeaders) {
        headers = newHeaders;
    }

    public static CountDownLatch getLatch() {
        return latch;
    }

    public static void setLatch(CountDownLatch newLatch) {
        latch = newLatch;
    }

    public static String getResult() {
        return result;
    }

    public static void setResult(String newResult) {
        result = newResult;
    }

    public static HttpResponseStatus getHttpStatus() {
        return httpStatus;
    }

    public static void setHttpStatus(HttpResponseStatus newStatus) {
        httpStatus = newStatus;
    }

    public static void unregisterModel(
            Channel channel, String modelName, String version, boolean syncChannel)
            throws InterruptedException {
        String requestURL = "/models/" + modelName;
        if (version != null) {
            requestURL += "/" + version;
        }

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, requestURL);
        if (syncChannel) {
            channel.writeAndFlush(req).sync();
            channel.closeFuture().sync();
        } else {
            channel.writeAndFlush(req);
        }
    }

    public static void registerModel(
            Channel channel,
            String url,
            String modelName,
            boolean withInitialWorkers,
            boolean syncChannel)
            throws InterruptedException {
        String requestURL = "/models?url=" + url + "&model_name=" + modelName + "&runtime=python";
        if (withInitialWorkers) {
            requestURL += "&initial_workers=1&synchronous=true";
        }

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, requestURL);
        if (syncChannel) {
            channel.writeAndFlush(req).sync();
            channel.closeFuture().sync();
        } else {
            channel.writeAndFlush(req);
        }
    }

    public static void scaleModel(
            Channel channel, String modelName, String version, int minWorker, boolean sync) {
        String requestURL = "/models/" + modelName;

        if (version != null) {
            requestURL += "/" + version;
        }

        requestURL += "?min_worker=" + minWorker;

        if (sync) {
            requestURL += "&synchronous=true";
        }

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        channel.writeAndFlush(req);
    }

    public static void getRoot(Channel channel) throws InterruptedException {
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/");
        channel.writeAndFlush(req).sync();
    }

    public static void getApiDescription(Channel channel) {
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/api-description");
        channel.writeAndFlush(req);
    }

    public static void describeModelApi(Channel channel, String modelName) {
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/" + modelName);
        channel.writeAndFlush(req);
    }

    public static void describeModel(Channel channel, String modelName, String version) {
        String requestURL = "/models/" + modelName;
        if (version != null) {
            requestURL += "/" + version;
        }

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, requestURL);
        channel.writeAndFlush(req);
    }

    public static void listModels(Channel channel) {
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models?limit=200&nextPageToken=X");
        channel.writeAndFlush(req);
    }

    public static void setDefault(Channel channel, String modelName, String defaultVersion) {
        String requestURL = "/models/" + modelName + "/" + defaultVersion + "/set-default";

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, requestURL);
        channel.writeAndFlush(req);
    }

    public static void ping(ConfigManager configManager) throws InterruptedException {
        Channel channel = TestUtils.getInferenceChannel(configManager);
        TestUtils.setResult(null);
        TestUtils.setLatch(new CountDownLatch(1));
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/ping");
        channel.writeAndFlush(req);
    }

    public static Channel connect(boolean management, ConfigManager configManager) {
        return connect(management, configManager, 120);
    }

    public static Channel connect(
            boolean management, ConfigManager configManager, int readTimeOut) {
        Logger logger = LoggerFactory.getLogger(ModelServerTest.class);

        final Connector connector = configManager.getListener(management);
        try {
            Bootstrap b = new Bootstrap();
            final SslContext sslCtx =
                    SslContextBuilder.forClient()
                            .trustManager(InsecureTrustManagerFactory.INSTANCE)
                            .build();
            b.group(Connector.newEventLoopGroup(1))
                    .channel(connector.getClientChannel())
                    .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 10000)
                    .handler(
                            new ChannelInitializer<Channel>() {
                                @Override
                                public void initChannel(Channel ch) {
                                    ChannelPipeline p = ch.pipeline();
                                    if (connector.isSsl()) {
                                        p.addLast(sslCtx.newHandler(ch.alloc()));
                                    }
                                    p.addLast(new ReadTimeoutHandler(readTimeOut));
                                    p.addLast(new HttpClientCodec());
                                    p.addLast(new HttpContentDecompressor());
                                    p.addLast(new ChunkedWriteHandler());
                                    p.addLast(new HttpObjectAggregator(6553600));
                                    p.addLast(new TestHandler());
                                }
                            });

            return b.connect(connector.getSocketAddress()).sync().channel();
        } catch (Throwable t) {
            logger.warn("Connect error.", t);
        }
        return null;
    }

    public static Channel getInferenceChannel(ConfigManager configManager)
            throws InterruptedException {
        return getChannel(false, configManager);
    }

    public static Channel getManagementChannel(ConfigManager configManager)
            throws InterruptedException {
        return getChannel(true, configManager);
    }

    private static Channel getChannel(boolean isManagementChannel, ConfigManager configManager)
            throws InterruptedException {
        if (isManagementChannel && managementChannel != null && managementChannel.isActive()) {
            return managementChannel;
        } else if (!isManagementChannel
                && inferenceChannel != null
                && inferenceChannel.isActive()) {
            return inferenceChannel;
        } else {
            Channel channel = null;
            if (channel == null) {
                for (int i = 0; i < 5; ++i) {
                    channel = TestUtils.connect(isManagementChannel, configManager);
                    if (channel != null) {
                        break;
                    }
                    Thread.sleep(100);
                }
            }
            if (isManagementChannel) {
                managementChannel = channel;
            } else {
                inferenceChannel = channel;
            }
            return channel;
        }
    }

    public static void closeChannels() throws InterruptedException {
        if (managementChannel != null) {
            managementChannel.closeFuture().sync();
        }
        if (inferenceChannel != null) {
            inferenceChannel.closeFuture().sync();
        }
    }

    @ChannelHandler.Sharable
    private static class TestHandler extends SimpleChannelInboundHandler<FullHttpResponse> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, FullHttpResponse msg) {
            httpStatus = msg.status();
            result = msg.content().toString(StandardCharsets.UTF_8);
            headers = msg.headers();
            latch.countDown();
        }

        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            Logger logger = LoggerFactory.getLogger(TestHandler.class);
            logger.error("Unknown exception", cause);
            ctx.close();
            latch.countDown();
        }
    }
}
