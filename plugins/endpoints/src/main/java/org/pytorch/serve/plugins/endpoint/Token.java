package org.pytorch.serve.plugins.endpoint;

// import java.util.Properties;
import com.google.gson.annotations.SerializedName;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.Base64;
import java.util.concurrent.TimeUnit;
import org.pytorch.serve.servingsdk.Context;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;
import org.pytorch.serve.servingsdk.http.Request;
import org.pytorch.serve.servingsdk.http.Response;

@Endpoint(
        urlPattern = "token",
        endpointType = EndpointTypes.INFERENCE,
        description = "Execution parameters endpoint")
public class Token extends ModelServerEndpoint {
    private boolean managementFlag;
    private String managementToken;

    @Override
    public void doGet(Request req, Response rsp, Context ctx) throws IOException {
        // Properties prop = ctx.getConfig();
        TokenResponse r = new TokenResponse();
        if (!managementFlag) {
            managementFlag = true;
            r.setKey();
            String output = "{\n\t\"Manager Key\": " + r.getKey() + "\n}\n";
            rsp.getOutputStream().write(output.getBytes(StandardCharsets.UTF_8));
            managementToken = r.getKey();
        }
        String test = "";
        if (r.keyFile(managementToken)) {
            test = "{\n\t\"File Updated\": successfully \n}\n";
        } else {
            test = "{\n\t\"File\": failed \n}\n";
        }
        rsp.getOutputStream().write(test.getBytes(StandardCharsets.UTF_8));
    }

    /** Response for Model server endpoint */
    public static class TokenResponse {
        private SecureRandom secureRandom = new SecureRandom();
        private Base64.Encoder baseEncoder = Base64.getUrlEncoder();

        @SerializedName("Key")
        private String key;

        @SerializedName("TokenExpiration")
        private Instant tokenExpiration;

        public TokenResponse() {
            key = "test 2";
            tokenExpiration = Instant.now();
        }

        public String getKey() {
            return key;
        }

        public Instant getTokenExpiration() {
            return tokenExpiration;
        }

        public void setKey() {
            key = generateKey();
        }

        public boolean keyFile(String managementToken) throws IOException {
            String fileSeparator = System.getProperty("file.separator");
            String fileData = " ";
            // Will change to get file path rather then being set defaulty
            String absoluteFilePath =
                    fileSeparator
                            + "home"
                            + fileSeparator
                            + "ubuntu"
                            + fileSeparator
                            + "serve/key_file.txt";
            File file = new File(absoluteFilePath);

            if (!file.createNewFile() && !file.exists()) {
                return false;
            }
            fileData =
                    "Management Key: "
                            + managementToken
                            + "\n"
                            + "Inference Key: "
                            + generateKey()
                            + " --- Expiration time: "
                            + tokenExpiration.toString()
                            + "\n";
            Files.write(Paths.get("key_file.txt"), fileData.getBytes());
            return true;
        }

        public String generateKey() {
            byte[] randomBytes = new byte[6];
            secureRandom.nextBytes(randomBytes);
            setTokenExpiration();
            return baseEncoder.encodeToString(randomBytes);
        }

        public void setTokenExpiration() {
            Integer time = 3;
            tokenExpiration = Instant.now().plusSeconds(TimeUnit.MINUTES.toSeconds(time));
        }
    }
}
