package org.pytorch.serve.util;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
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
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TokenAuthorization {
    private static String apiKey;
    private static String managementKey;
    private static String inferenceKey;
    private static Instant managementExpirationTimeMinutes;
    private static Instant inferenceExpirationTimeMinutes;
    private static Boolean tokenAuthEnabled;
    private static String keyFilePath;
    private static final SecureRandom secureRandom = new SecureRandom();
    private static final Base64.Encoder baseEncoder = Base64.getUrlEncoder();
    private static final Pattern bearerTokenHeaderPattern = Pattern.compile("^Bearer\\s+(\\S+)$");
    private static final Logger logger = LoggerFactory.getLogger(TokenAuthorization.class);

    public enum TokenType {
        INFERENCE,
        MANAGEMENT,
        TOKEN_API
    }

    public static void init() {
        if (ConfigManager.getInstance().getDisableTokenAuthorization()) {
            tokenAuthEnabled = false;
            return;
        }

        tokenAuthEnabled = true;
        apiKey = generateKey();
        keyFilePath = Paths.get(System.getProperty("user.dir"), "key_file.json").toString();

        try {
            if (generateKeyFile(TokenType.TOKEN_API)) {
                String loggingMessage =
                        "\n######\n"
                                + "TorchServe now enforces token authorization by default.\n"
                                + "This requires the correct token to be provided when calling an API.\n"
                                + "Key file located at "
                                + keyFilePath
                                + "\nCheck token authorization documenation for information: https://github.com/pytorch/serve/blob/master/docs/token_authorization_api.md \n"
                                + "######\n";
                logger.info(loggingMessage);
            }
        } catch (IOException e) {
            e.printStackTrace();
            logger.error("Token Authorization setup unsuccessful");
            throw new IllegalStateException("Token Authorization setup unsuccessful", e);
        }
    }

    public static Boolean isEnabled() {
        return tokenAuthEnabled;
    }

    public static String updateKeyFile(TokenType tokenType) throws IOException {
        String status = "";

        switch (tokenType) {
            case MANAGEMENT:
                generateKeyFile(TokenType.MANAGEMENT);
                break;
            case INFERENCE:
                generateKeyFile(TokenType.INFERENCE);
                break;
            default:
                status = "{\n\t\"Error\": " + tokenType + "\n}\n";
        }

        return status;
    }

    public static boolean checkTokenAuthorization(String token, TokenType tokenType) {
        String key;
        Instant expiration;
        switch (tokenType) {
            case TOKEN_API:
                key = apiKey;
                expiration = null;
                break;
            case MANAGEMENT:
                key = managementKey;
                expiration = managementExpirationTimeMinutes;
                break;
            default:
                key = inferenceKey;
                expiration = inferenceExpirationTimeMinutes;
        }

        if (token.equals(key)) {
            if (expiration != null && isTokenExpired(expiration)) {
                return false;
            }
        } else {
            return false;
        }
        return true;
    }

    public static String parseTokenFromBearerTokenHeader(String bearerTokenHeader) {
        String token = "";
        Matcher matcher = bearerTokenHeaderPattern.matcher(bearerTokenHeader);
        if (matcher.matches()) {
            token = matcher.group(1);
        }

        return token;
    }

    private static String generateKey() {
        byte[] randomBytes = new byte[6];
        secureRandom.nextBytes(randomBytes);
        return baseEncoder.encodeToString(randomBytes);
    }

    private static Instant generateTokenExpiration() {
        long secondsToAdd = (long) (ConfigManager.getInstance().getTimeToExpiration() * 60);
        return Instant.now().plusSeconds(secondsToAdd);
    }

    private static boolean generateKeyFile(TokenType tokenType) throws IOException {
        File file = new File(keyFilePath);
        if (!file.createNewFile() && !file.exists()) {
            return false;
        }
        switch (tokenType) {
            case MANAGEMENT:
                managementKey = generateKey();
                managementExpirationTimeMinutes = generateTokenExpiration();
                break;
            case INFERENCE:
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
                Paths.get(keyFilePath),
                new GsonBuilder()
                        .setPrettyPrinting()
                        .create()
                        .toJson(parentObject)
                        .getBytes(StandardCharsets.UTF_8));

        if (!setFilePermissions()) {
            try {
                Files.delete(Paths.get(keyFilePath));
            } catch (IOException e) {
                return false;
            }
            return false;
        }
        return true;
    }

    private static boolean setFilePermissions() {
        Path path = Paths.get(keyFilePath);
        try {
            Set<PosixFilePermission> permissions = PosixFilePermissions.fromString("rw-------");
            Files.setPosixFilePermissions(path, permissions);
        } catch (Exception e) {
            return false;
        }
        return true;
    }

    private static boolean isTokenExpired(Instant expirationTime) {
        return !(Instant.now().isBefore(expirationTime));
    }
}
