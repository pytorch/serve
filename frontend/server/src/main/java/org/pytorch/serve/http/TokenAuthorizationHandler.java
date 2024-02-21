package org.pytorch.serve.http;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.lang.reflect.*;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.InvalidKeyException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.TokenType;
import org.pytorch.serve.wlm.WorkerInitializationException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class handling token check for all inbound HTTP requests
 *
 * <p>This class //
 */
public class TokenAuthorizationHandler extends HttpRequestHandlerChain {

    private static final Logger logger = LoggerFactory.getLogger(TokenAuthorizationHandler.class);
    private static TokenType tokenType;
    private static Boolean tokenEnabled = false;
    private static Class<?> tokenClass;
    private static Object tokenObject;
    private static Double timeToExpirationMinutes = 60.0;

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
        if (tokenEnabled) {
            if (tokenType == TokenType.MANAGEMENT) {
                if (req.toString().contains("/token")) {
                    checkTokenAuthorization(req, "token");
                } else {
                    checkTokenAuthorization(req, "management");
                }
            } else if (tokenType == TokenType.INFERENCE) {
                checkTokenAuthorization(req, "inference");
            }
        }
        chain.handleRequest(ctx, req, decoder, segments);
    }

    public static void setupTokenClass() {
        try {
            tokenClass = Class.forName("org.pytorch.serve.plugins.endpoint.Token");
            tokenObject = tokenClass.getDeclaredConstructor().newInstance();
            Method method = tokenClass.getMethod("setTime", Double.class);
            Double time = ConfigManager.getInstance().getTimeToExpiration();
            if (time != 0.0) {
                timeToExpirationMinutes = time;
            }
            method.invoke(tokenObject, timeToExpirationMinutes);
            method = tokenClass.getMethod("generateKeyFile", String.class);
            if ((boolean) method.invoke(tokenObject, "token")) {
                logger.info("TOKEN CLASS IMPORTED SUCCESSFULLY");
            }
        } catch (NoSuchMethodException
                | IllegalAccessException
                | InstantiationException
                | InvocationTargetException
                | ClassNotFoundException e) {
            e.printStackTrace();
            logger.error("TOKEN CLASS IMPORTED UNSUCCESSFULLY");
            throw new IllegalStateException("Unable to import token class", e);
        }
        tokenEnabled = true;
    }

    private void checkTokenAuthorization(FullHttpRequest req, String type) throws ModelException {

        try {
            Method method =
                    tokenClass.getMethod(
                            "checkTokenAuthorization",
                            io.netty.handler.codec.http.FullHttpRequest.class,
                            String.class);
            boolean result = (boolean) (method.invoke(tokenObject, req, type));
            if (!result) {
                throw new InvalidKeyException(
                        "Token Authentication failed. Token either incorrect, expired, or not provided correctly");
            }
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            e.printStackTrace();
            throw new InvalidKeyException(
                    "Token Authentication failed. Token either incorrect, expired, or not provided correctly");
        }
    }
}
