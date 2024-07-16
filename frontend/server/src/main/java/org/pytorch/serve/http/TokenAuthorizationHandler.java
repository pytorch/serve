package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.util.List;
import java.util.Map;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.InvalidKeyException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.util.NettyUtils;
import org.pytorch.serve.util.TokenAuthorization;
import org.pytorch.serve.util.TokenAuthorization.TokenType;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling token check for all inbound HTTP requests
 *
 * <p>This class //
 */
public class TokenAuthorizationHandler extends HttpRequestHandlerChain {
    private TokenType tokenType;
    private static final Logger logger = LoggerFactory.getLogger(TokenAuthorizationHandler.class);

    /** Creates a new {@code InferenceRequestHandler} instance. */
    public TokenAuthorizationHandler(TokenType type) {
        tokenType = type;
    }

    @Override
    public void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException, DownloadArchiveException, WorkflowException,
                    WorkerInitializationException {
        if (TokenAuthorization.isEnabled()) {
            if (tokenType == TokenType.MANAGEMENT) {
                if (req.toString().contains("/token")) {
                    try {
                        checkTokenAuthorization(req, TokenType.TOKEN_API);
                        String queryResponse = parseQuery(req);
                        String resp =
                                TokenAuthorization.updateKeyFile(
                                        TokenType.valueOf(queryResponse.toUpperCase()));
                        NettyUtils.sendJsonResponse(ctx, resp);
                        return;
                    } catch (Exception e) {
                        logger.error("Failed to update key file");
                        throw new InvalidKeyException(
                                "Token Authentication failed. Token either incorrect, expired, or not provided correctly");
                    }
                } else {
                    checkTokenAuthorization(req, TokenType.MANAGEMENT);
                }
            } else if (tokenType == TokenType.INFERENCE) {
                checkTokenAuthorization(req, TokenType.INFERENCE);
            }
        }
        chain.handleRequest(ctx, req, decoder, segments);
    }

    private void checkTokenAuthorization(FullHttpRequest req, TokenType tokenType)
            throws ModelException {
        String tokenBearer = req.headers().get("Authorization");
        if (tokenBearer == null) {
            throw new InvalidKeyException(
                    "Token Authorization failed. Token either incorrect, expired, or not provided correctly");
        }
        String token = TokenAuthorization.parseTokenFromBearerTokenHeader(tokenBearer);
        if (!TokenAuthorization.checkTokenAuthorization(token, tokenType)) {
            throw new InvalidKeyException(
                    "Token Authorization failed. Token either incorrect, expired, or not provided correctly");
        }
    }

    // parses query and either returns management/inference or a wrong type error
    private String parseQuery(FullHttpRequest req) {
        QueryStringDecoder decoder = new QueryStringDecoder(req.uri());
        Map<String, List<String>> parameters = decoder.parameters();
        List<String> values = parameters.get("type");
        if (values != null && !values.isEmpty()) {
            if ("management".equals(values.get(0)) || "inference".equals(values.get(0))) {
                return values.get(0);
            } else {
                return "WRONG TYPE";
            }
        }
        return "NO TYPE PROVIDED";
    }
}
