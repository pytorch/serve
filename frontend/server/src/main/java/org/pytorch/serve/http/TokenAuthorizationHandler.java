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
 * A class handling inbound HTTP requests to the inference API.
 *
 * <p>This class //
 */
public class TokenAuthorizationHandler extends HttpRequestHandlerChain {

    private static final Logger logger = LoggerFactory.getLogger(TokenAuthorizationHandler.class);
    private static TokenType tokenType;
    private static Boolean tokenEnabled = false;
    private static Class<?> tokenClass;
    private static Object tokenObject;
    private static Integer timeToExpirationMinutes = 60;

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
        ConfigManager configManager = ConfigManager.getInstance();
        if (tokenType == TokenType.MANAGEMENT) {
            if (req.toString().contains("/token")) {
                checkTokenAuthorization(req, 0);
            } else {
                checkTokenAuthorization(req, 1);
            }
        } else if (tokenType == TokenType.INFERENCE) {
            checkTokenAuthorization(req, 2);
        }
        chain.handleRequest(ctx, req, decoder, segments);
    }

    public static void setupTokenClass() {
        try {
            tokenClass = Class.forName("org.pytorch.serve.plugins.endpoint.Token");
            tokenObject = tokenClass.getDeclaredConstructor().newInstance();
            Method method = tokenClass.getMethod("setTime", Integer.class);
            Integer time = ConfigManager.getInstance().getTimeToExpiration();
            if (time == 0) {
                timeToExpirationMinutes = time;
            }
            method.invoke(tokenObject, timeToExpirationMinutes);
            method = tokenClass.getMethod("generateKeyFile", Integer.class);
            if ((boolean) method.invoke(tokenObject, 0)) {
                logger.info("TOKEN CLASS IMPORTED SUCCESSFULLY");
            }
        } catch (ClassNotFoundException e) {
            logger.error("TOKEN CLASS IMPORTED UNSUCCESSFULLY");
            e.printStackTrace();
            return;
        } catch (NoSuchMethodException
                | IllegalAccessException
                | InstantiationException
                | InvocationTargetException e) {
            e.printStackTrace();
            logger.error("TOKEN CLASS IMPORTED UNSUCCESSFULLY");
            return;
        }
        tokenEnabled = true;
    }

    private void checkTokenAuthorization(FullHttpRequest req, Integer type) throws ModelException {

        if (tokenEnabled) {
            try {
                Method method =
                        tokenClass.getMethod(
                                "checkTokenAuthorization",
                                io.netty.handler.codec.http.FullHttpRequest.class,
                                Integer.class);
                boolean result = (boolean) (method.invoke(tokenObject, req, type));
                if (!result) {
                    throw new InvalidKeyException(
                            "Token Authenticaation failed. Token either incorrect, expired, or not provided correctly");
                }
            } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
                e.printStackTrace();
                throw new InvalidKeyException(
                        "Token Authenticaation failed. Token either incorrect, expired, or not provided correctly");
            }
        }
    }
}
