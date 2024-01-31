package org.pytorch.serve;

import io.netty.channel.Channel;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.ssl.SslContext;
import org.pytorch.serve.http.ExtendedSSLHandler;
import org.pytorch.serve.http.HttpRequestHandler;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.http.InvalidRequestHandler;
import org.pytorch.serve.http.TokenAuthorizationHandler;
import org.pytorch.serve.http.api.rest.ApiDescriptionRequestHandler;
import org.pytorch.serve.http.api.rest.InferenceRequestHandler;
import org.pytorch.serve.http.api.rest.ManagementRequestHandler;
import org.pytorch.serve.http.api.rest.OpenInferenceProtocolRequestHandler;
import org.pytorch.serve.http.api.rest.PrometheusMetricsRequestHandler;
import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.TokenType;
import org.pytorch.serve.workflow.api.http.WorkflowInferenceRequestHandler;
import org.pytorch.serve.workflow.api.http.WorkflowMgmtRequestHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A special {@link io.netty.channel.ChannelInboundHandler} which offers an easy way to initialize a
 * {@link io.netty.channel.Channel} once it was registered to its {@link
 * io.netty.channel.EventLoop}.
 */
public class ServerInitializer extends ChannelInitializer<Channel> {

    private ConnectorType connectorType;
    private SslContext sslCtx;
    private static final Logger logger = LoggerFactory.getLogger(ServerInitializer.class);

    /**
     * Creates a new {@code HttpRequestHandler} instance.
     *
     * @param sslCtx null if SSL is not enabled
     * @param type true to initialize a management server instead of an API Server
     */
    public ServerInitializer(SslContext sslCtx, ConnectorType type) {
        this.sslCtx = sslCtx;
        this.connectorType = type;
    }

    /** {@inheritDoc} */
    @Override
    public void initChannel(Channel ch) {
        ChannelPipeline pipeline = ch.pipeline();
        HttpRequestHandlerChain apiDescriptionRequestHandler =
                new ApiDescriptionRequestHandler(connectorType);
        HttpRequestHandlerChain invalidRequestHandler = new InvalidRequestHandler();

        int maxRequestSize = ConfigManager.getInstance().getMaxRequestSize();
        if (sslCtx != null) {
            pipeline.addLast("ssl", new ExtendedSSLHandler(sslCtx, connectorType));
        }
        pipeline.addLast("http", new HttpServerCodec());
        pipeline.addLast("aggregator", new HttpObjectAggregator(maxRequestSize));

        HttpRequestHandlerChain httpRequestHandlerChain = apiDescriptionRequestHandler;
        if (ConnectorType.ALL.equals(connectorType)
                || ConnectorType.INFERENCE_CONNECTOR.equals(connectorType)) {
            httpRequestHandlerChain =
                    httpRequestHandlerChain.setNextHandler(
                            new TokenAuthorizationHandler(TokenType.INFERENCE));
            httpRequestHandlerChain =
                    httpRequestHandlerChain.setNextHandler(
                            new InferenceRequestHandler(
                                    PluginsManager.getInstance().getInferenceEndpoints()));
            httpRequestHandlerChain =
                    httpRequestHandlerChain.setNextHandler(new WorkflowInferenceRequestHandler());

            // Added OIP protocol with inference connector
            if (ConfigManager.getInstance().isOpenInferenceProtocol()) {
                logger.info("OIP added with handler chain");
                httpRequestHandlerChain =
                        httpRequestHandlerChain.setNextHandler(
                                new OpenInferenceProtocolRequestHandler());
            }
        }
        if (ConnectorType.ALL.equals(connectorType)
                || ConnectorType.MANAGEMENT_CONNECTOR.equals(connectorType)) {
            httpRequestHandlerChain =
                    httpRequestHandlerChain.setNextHandler(
                            new TokenAuthorizationHandler(TokenType.MANAGEMENT));
            httpRequestHandlerChain =
                    httpRequestHandlerChain.setNextHandler(
                            new ManagementRequestHandler(
                                    PluginsManager.getInstance().getManagementEndpoints()));
            httpRequestHandlerChain =
                    httpRequestHandlerChain.setNextHandler(new WorkflowMgmtRequestHandler());
        }
        if (ConfigManager.getInstance().isMetricApiEnable()
                        && ConnectorType.ALL.equals(connectorType)
                || ConnectorType.METRICS_CONNECTOR.equals(connectorType)) {
            httpRequestHandlerChain =
                    httpRequestHandlerChain.setNextHandler(new PrometheusMetricsRequestHandler());
        }
        httpRequestHandlerChain.setNextHandler(invalidRequestHandler);
        pipeline.addLast("handler", new HttpRequestHandler(apiDescriptionRequestHandler));
    }
}
