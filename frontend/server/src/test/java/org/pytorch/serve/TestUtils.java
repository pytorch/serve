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
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import java.util.regex.Pattern;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.NettyUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class TestUtils {

    static CountDownLatch latch;
    static HttpResponseStatus httpStatus;
    static byte[] content;
    static String result;
    static HttpHeaders headers;
    private static Channel inferenceChannel;
    private static Channel managementChannel;
    private static Channel metricsChannel;
    private static String tsInferLatencyPattern =
            "ts_inference_latency_microseconds\\{"
                    + "uuid=\"[\\w]{8}(-[\\w]{4}){3}-[\\w]{12}\","
                    + "model_name=\"%s\","
                    + "model_version=\"%s\",\\}\\s\\d+(\\.\\d+)";

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

    public static byte[] getContent() {
        return content;
    }

    public static void setContent(byte[] newContent) {
        content = newContent;
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

    public static void unregisterWorkflow(Channel channel, String workflowName, boolean syncChannel)
            throws InterruptedException {
        String requestURL = "/workflows/" + workflowName;

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

    public static void registerModel(
            Channel channel,
            String url,
            String modelName,
            boolean withInitialWorkers,
            boolean syncChannel,
            int batchSize,
            int maxBatchDelay)
            throws InterruptedException {
        String requestURL =
                "/models?url="
                        + url
                        + "&model_name="
                        + modelName
                        + "&runtime=python"
                        + "&batch_size="
                        + batchSize
                        + "&max_batch_delay="
                        + maxBatchDelay;
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

    public static void registerWorkflow(
            Channel channel, String url, String workflowName, boolean syncChannel)
            throws InterruptedException {
        String requestURL = "/workflows?url=" + url + "&workflow_name=" + workflowName;

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

    public static void describeWorkflow(Channel channel, String workflowName) {
        String requestURL = "/workflows/" + workflowName;

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

    public static void listWorkflow(Channel channel) {
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.GET,
                        "/workflows?limit=200&nextPageToken=X");
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

    public static Channel connect(ConnectorType connectorType, ConfigManager configManager) {
        return connect(connectorType, configManager, 240);
    }

    public static Channel connect(
            ConnectorType connectorType, ConfigManager configManager, int readTimeOut) {
        Logger logger = LoggerFactory.getLogger(TestUtils.class);

        final Connector connector = configManager.getListener(connectorType);
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
        return getChannel(ConnectorType.INFERENCE_CONNECTOR, configManager);
    }

    public static Channel getManagementChannel(ConfigManager configManager)
            throws InterruptedException {
        return getChannel(ConnectorType.MANAGEMENT_CONNECTOR, configManager);
    }

    public static Channel getMetricsChannel(ConfigManager configManager)
            throws InterruptedException {
        return getChannel(ConnectorType.METRICS_CONNECTOR, configManager);
    }

    private static Channel getChannel(ConnectorType connectorType, ConfigManager configManager)
            throws InterruptedException {
        if (ConnectorType.MANAGEMENT_CONNECTOR.equals(connectorType)
                && managementChannel != null
                && managementChannel.isActive()) {
            return managementChannel;
        }
        if (ConnectorType.INFERENCE_CONNECTOR.equals(connectorType)
                && inferenceChannel != null
                && inferenceChannel.isActive()) {
            return inferenceChannel;
        }
        if (ConnectorType.METRICS_CONNECTOR.equals(connectorType)
                && metricsChannel != null
                && metricsChannel.isActive()) {
            return metricsChannel;
        }
        Channel channel = null;
        for (int i = 0; i < 5; ++i) {
            channel = TestUtils.connect(connectorType, configManager);
            if (channel != null) {
                break;
            }
            Thread.sleep(100);
        }
        switch (connectorType) {
            case MANAGEMENT_CONNECTOR:
                managementChannel = channel;
                break;
            case METRICS_CONNECTOR:
                metricsChannel = channel;
                break;
            default:
                inferenceChannel = channel;
        }
        return channel;
    }

    public static void closeChannels() throws InterruptedException {
        if (managementChannel != null) {
            managementChannel.closeFuture().sync();
        }
        if (inferenceChannel != null) {
            inferenceChannel.closeFuture().sync();
        }
        if (metricsChannel != null) {
            metricsChannel.closeFuture().sync();
        }
    }

    public static Pattern getTSInferLatencyMatcher(String modelName, String modelVersion) {
        modelVersion = modelVersion == null ? "default" : modelVersion;
        return Pattern.compile(
                String.format(TestUtils.tsInferLatencyPattern, modelName, modelVersion));
    }

    public static void setConfiguration(ConfigManager configManager, String key, String val)
            throws NoSuchFieldException, IllegalAccessException {
        Field f = configManager.getClass().getDeclaredField("prop");
        f.setAccessible(true);
        Properties p = (Properties) f.get(configManager);
        p.setProperty(key, val);
    }

    @ChannelHandler.Sharable
    private static class TestHandler extends SimpleChannelInboundHandler<FullHttpResponse> {

        @Override
        public void channelRead0(ChannelHandlerContext ctx, FullHttpResponse msg) {
            httpStatus = msg.status();
            content = NettyUtils.getBytes(msg.content());
            result = new String(content, StandardCharsets.UTF_8);
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
