package org.pytorch.serve.http;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.InvalidKeyException;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.workflow.WorkflowException;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.NettyUtils;
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
    private static Token tokenClass;
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
                    try {
                        checkTokenAuthorization(req, "token");
                        String resp = tokenClass.updateKeyFile(req);
                        NettyUtils.sendJsonResponse(ctx, resp);
                        return;
                    } catch (Exception e) {
                        logger.error("TOKEN CLASS UPDATED UNSUCCESSFULLY");
                        throw new InvalidKeyException(
                                "Token Authentication failed. Token either incorrect, expired, or not provided correctly");
                    }
                } else {
                    checkTokenAuthorization(req, "management");
                }
            } else if (tokenType == TokenType.INFERENCE) {
                checkTokenAuthorization(req, "inference");
            }
        } else {
            if (tokenType == TokenType.MANAGEMENT && req.toString().contains("/token")) {
                throw new ResourceNotFoundException();
            }
        }
        chain.handleRequest(ctx, req, decoder, segments);
    }

    public static void setupTokenClass() {
        try {
            tokenClass = new Token();
            Double time = ConfigManager.getInstance().getTimeToExpiration();
            String home = ConfigManager.getInstance().getModelServerHome();
            tokenClass.setFilePath(home);
            if (time != 0.0) {
                timeToExpirationMinutes = time;
            }
            tokenClass.setTime(timeToExpirationMinutes);
            if (tokenClass.generateKeyFile("token")) {
                logger.info("Token Authorization Enabled");
            }
        } catch (Exception e) {
            e.printStackTrace();
            logger.error("TOKEN CLASS IMPORTED UNSUCCESSFULLY");
            throw new IllegalStateException("Unable to import token class", e);
        }
        tokenEnabled = true;
    }

    private void checkTokenAuthorization(FullHttpRequest req, String type) throws ModelException {

        try {
            boolean result = tokenClass.checkTokenAuthorization(req, type);
            if (!result) {
                throw new InvalidKeyException(
                        "Token Authentication failed. Token either incorrect, expired, or not provided correctly");
            }
        } catch (Exception e) {
            throw new InvalidKeyException(
                    "Token Authentication failed. Token either incorrect, expired, or not provided correctly");
        }
    }
}

class Token {
    private static String apiKey;
    private static String managementKey;
    private static String inferenceKey;
    private static Instant managementExpirationTimeMinutes;
    private static Instant inferenceExpirationTimeMinutes;
    private static Double timeToExpirationMinutes;
    private SecureRandom secureRandom = new SecureRandom();
    private Base64.Encoder baseEncoder = Base64.getUrlEncoder();
    private String fileName = "key_file.json";
    private String filePath = "";

    public String updateKeyFile(FullHttpRequest req) throws IOException {
        String queryResponse = parseQuery(req);
        String test = "";
        if ("management".equals(queryResponse)) {
            generateKeyFile("management");
        } else if ("inference".equals(queryResponse)) {
            generateKeyFile("inference");
        } else {
            test = "{\n\t\"Error\": " + queryResponse + "\n}\n";
        }
        return test;
    }

    // parses query and either returns management/inference or a wrong type error
    public String parseQuery(FullHttpRequest req) {
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

    public String generateKey() {
        byte[] randomBytes = new byte[6];
        secureRandom.nextBytes(randomBytes);
        return baseEncoder.encodeToString(randomBytes);
    }

    public Instant generateTokenExpiration() {
        long secondsToAdd = (long) (timeToExpirationMinutes * 60);
        return Instant.now().plusSeconds(secondsToAdd);
    }

    public void setFilePath(String path) {
        filePath = path;
    }

    // generates a key file with new keys depending on the parameter provided
    public boolean generateKeyFile(String type) throws IOException {
        String userDirectory = filePath + "/" + fileName;
        File file = new File(userDirectory);
        if (!file.createNewFile() && !file.exists()) {
            return false;
        }
        if (apiKey == null) {
            apiKey = generateKey();
        }
        switch (type) {
            case "management":
                managementKey = generateKey();
                managementExpirationTimeMinutes = generateTokenExpiration();
                break;
            case "inference":
                inferenceKey = generateKey();
                inferenceExpirationTimeMinutes = generateTokenExpiration();
                break;
            default:
                managementKey = generateKey();
                inferenceKey = generateKey();
                inferenceExpirationTimeMinutes = generateTokenExpiration();
                managementExpirationTimeMinutes = generateTokenExpiration();
        }

        JsonObject parentObject = new JsonObject();

        JsonObject managementObject = new JsonObject();
        managementObject.addProperty("key", managementKey);
        managementObject.addProperty("expiration time", managementExpirationTimeMinutes.toString());
        parentObject.add("management", managementObject);

        JsonObject inferenceObject = new JsonObject();
        inferenceObject.addProperty("key", inferenceKey);
        inferenceObject.addProperty("expiration time", inferenceExpirationTimeMinutes.toString());
        parentObject.add("inference", inferenceObject);

        JsonObject apiObject = new JsonObject();
        apiObject.addProperty("key", apiKey);
        parentObject.add("API", apiObject);

        Files.write(
                Paths.get(fileName),
                new GsonBuilder()
                        .setPrettyPrinting()
                        .create()
                        .toJson(parentObject)
                        .getBytes(StandardCharsets.UTF_8));

        if (!setFilePermissions()) {
            try {
                Files.delete(Paths.get(fileName));
            } catch (IOException e) {
                return false;
            }
            return false;
        }
        return true;
    }

    public boolean setFilePermissions() {
        Path path = Paths.get(fileName);
        try {
            Set<PosixFilePermission> permissions = PosixFilePermissions.fromString("rw-------");
            Files.setPosixFilePermissions(path, permissions);
        } catch (Exception e) {
            return false;
        }
        return true;
    }

    // checks the token provided in the http with the saved keys depening on parameters
    public boolean checkTokenAuthorization(FullHttpRequest req, String type) {
        String key;
        Instant expiration;
        switch (type) {
            case "token":
                key = apiKey;
                expiration = null;
                break;
            case "management":
                key = managementKey;
                expiration = managementExpirationTimeMinutes;
                break;
            default:
                key = inferenceKey;
                expiration = inferenceExpirationTimeMinutes;
        }

        String tokenBearer = req.headers().get("Authorization");
        if (tokenBearer == null) {
            return false;
        }
        String[] arrOfStr = tokenBearer.split(" ", 2);
        if (arrOfStr.length == 1) {
            return false;
        }
        String token = arrOfStr[1];

        if (token.equals(key)) {
            if (expiration != null && isTokenExpired(expiration)) {
                return false;
            }
        } else {
            return false;
        }
        return true;
    }

    public boolean isTokenExpired(Instant expirationTime) {
        return !(Instant.now().isBefore(expirationTime));
    }

    public String getManagementKey() {
        return managementKey;
    }

    public String getInferenceKey() {
        return inferenceKey;
    }

    public String getKey() {
        return apiKey;
    }

    public Instant getInferenceExpirationTime() {
        return inferenceExpirationTimeMinutes;
    }

    public Instant getManagementExpirationTime() {
        return managementExpirationTimeMinutes;
    }

    public void setTime(Double time) {
        timeToExpirationMinutes = time;
    }
}
