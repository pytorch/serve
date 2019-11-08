/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.pytorch.serve;

import io.netty.channel.Channel;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.ssl.SslContext;
import org.pytorch.serve.http.ApiDescriptionRequestHandler;
import org.pytorch.serve.http.HttpRequestHandler;
import org.pytorch.serve.http.HttpRequestHandlerChain;
import org.pytorch.serve.http.InferenceRequestHandler;
import org.pytorch.serve.http.InvalidRequestHandler;
import org.pytorch.serve.http.ManagementRequestHandler;
import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.ConnectorType;

/**
 * A special {@link io.netty.channel.ChannelInboundHandler} which offers an easy way to initialize a
 * {@link io.netty.channel.Channel} once it was registered to its {@link
 * io.netty.channel.EventLoop}.
 */
public class ServerInitializer extends ChannelInitializer<Channel> {

    private ConnectorType connectorType;
    private SslContext sslCtx;

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
            pipeline.addLast("ssl", sslCtx.newHandler(ch.alloc()));
        }
        pipeline.addLast("http", new HttpServerCodec());
        pipeline.addLast("aggregator", new HttpObjectAggregator(maxRequestSize));

        HttpRequestHandlerChain httpRequestHandlerChain = apiDescriptionRequestHandler;
        if (ConnectorType.BOTH.equals(connectorType)
                || ConnectorType.INFERENCE_CONNECTOR.equals(connectorType)) {
            httpRequestHandlerChain =
                    httpRequestHandlerChain.setNextHandler(
                            new InferenceRequestHandler(
                                    PluginsManager.getInstance().getInferenceEndpoints()));
        }
        if (ConnectorType.BOTH.equals(connectorType)
                || ConnectorType.MANAGEMENT_CONNECTOR.equals(connectorType)) {
            httpRequestHandlerChain =
                    httpRequestHandlerChain.setNextHandler(
                            new ManagementRequestHandler(
                                    PluginsManager.getInstance().getManagementEndpoints()));
        }
        httpRequestHandlerChain.setNextHandler(invalidRequestHandler);
        pipeline.addLast("handler", new HttpRequestHandler(apiDescriptionRequestHandler));
    }
}
