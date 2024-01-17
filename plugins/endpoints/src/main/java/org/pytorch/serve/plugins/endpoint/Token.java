package org.pytorch.serve.plugins.endpoint;

// import java.util.Properties;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.pytorch.serve.servingsdk.Context;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.http.Response;

@Endpoint(
        urlPattern = "token",
        endpointType = EndpointTypes.MANAGEMENT,
        description = "Token authentication endpoint")
public class Token extends ModelServerEndpoint {
    private static String apiKey;
    private static String managementKey;
    private static String inferenceKey;
    private static Instant managementExpirationTime;
    private static Instant inferenceExpirationTime;
    private static Integer timeToExpiration = 30;
    private SecureRandom secureRandom = new SecureRandom();
    private Base64.Encoder baseEncoder = Base64.getUrlEncoder();

    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        String queryResponse = parseQuery(req);
        String test = "";
        if ("management".equals(queryResponse)) {
            generateKeyFile(1);
        } else if ("inference".equals(queryResponse)) {
            generateKeyFile(2);
        } else {
            test = "{\n\t\"Error\": " + queryResponse + "\n}\n";
        }
        rsp.getOutputStream().write(test.getBytes(StandardCharsets.UTF_8));
    }

    // parses query and either returns "management"/"inference" or a wrong type error
    public String parseQuery(Request req) {
        QueryStringDecoder decoder = new QueryStringDecoder(req.getRequestURI());
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

    public Instant generateTokenExpiration(Integer time) {
        return Instant.now().plusSeconds(TimeUnit.MINUTES.toSeconds(time));
    }

    // generates a key file with new keys depending on the parameter provided
    // 0: generates all 3 keys
    // 1: generates management key and keeps other 2 the same
    // 2: generates inference key and keeps other 2 the same
    public boolean generateKeyFile(Integer keyCase) throws IOException {
        String fileData = " ";
        String userDirectory = System.getProperty("user.dir") + "/key_file.txt";
        File file = new File(userDirectory);
        if (!file.createNewFile() && !file.exists()) {
            return false;
        }
        if (apiKey == null) {
            apiKey = generateKey();
        }
        switch (keyCase) {
            case 1:
                managementKey = generateKey();
                managementExpirationTime = generateTokenExpiration(timeToExpiration);
                break;
            case 2:
                inferenceKey = generateKey();
                inferenceExpirationTime = generateTokenExpiration(timeToExpiration);
                break;
            default:
                managementKey = generateKey();
                inferenceKey = generateKey();
                inferenceExpirationTime = generateTokenExpiration(timeToExpiration);
                managementExpirationTime = generateTokenExpiration(timeToExpiration);
        }

        fileData =
                "Management Key: "
                        + managementKey
                        + " --- Expiration time: "
                        + managementExpirationTime
                        + "\nInference Key: "
                        + inferenceKey
                        + " --- Expiration time: "
                        + inferenceExpirationTime
                        + "\nAPI Key: "
                        + apiKey
                        + "\n";
        Files.write(Paths.get("key_file.txt"), fileData.getBytes());
        return true;
    }

    // checks the token provided in the http with the saved keys depening on parameters
    public boolean checkTokenAuthorization(FullHttpRequest req, Integer keyCase) {
        String key;
        Instant expiration;
        switch (keyCase) {
            case 0:
                key = apiKey;
                expiration = null;
                break;
            case 1:
                key = managementKey;
                expiration = managementExpirationTime;
                break;
            default:
                key = inferenceKey;
                expiration = inferenceExpirationTime;
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
        return inferenceExpirationTime;
    }

    public Instant getManagementExpirationTime() {
        return managementExpirationTime;
    }

    public void setTime(Integer time) {
        timeToExpiration = time;
    }
}
