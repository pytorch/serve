package org.pytorch.serve.http;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.ssl.OptionalSslHandler;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslHandler;
import java.util.List;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.NettyUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ExtendedSSLHandler extends OptionalSslHandler {
    private static final Logger logger = LoggerFactory.getLogger(ExtendedSSLHandler.class);
    /** the length of the ssl record header (in bytes) */
    private static final int SSL_RECORD_HEADER_LENGTH = 5;

    private ConnectorType connectorType;

    public ExtendedSSLHandler(SslContext sslContext, ConnectorType connectorType) {
        super(sslContext);
        this.connectorType = connectorType;
    }

    @Override
    protected void decode(ChannelHandlerContext context, ByteBuf in, List<Object> out)
            throws Exception {
        if (in.readableBytes() < SSL_RECORD_HEADER_LENGTH) {
            return;
        }
        ConfigManager configMgr = ConfigManager.getInstance();
        if (SslHandler.isEncrypted(in) || !configMgr.isSSLEnabled(connectorType)) {
            super.decode(context, in, out);
        } else {
            logger.error("Recieved HTTP request!");
            NettyUtils.sendJsonResponse(
                    context,
                    new StatusResponse(
                            "This TorchServe instance only accepts HTTPS requests",
                            HttpResponseStatus.FORBIDDEN.code()));
        }
    }
}
