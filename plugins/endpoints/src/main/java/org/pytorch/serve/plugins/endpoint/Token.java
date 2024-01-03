package org.pytorch.serve.plugins.endpoint;

// import java.util.Properties;
import com.google.gson.annotations.SerializedName;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
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
        TokenResponse r = new TokenResponse();
        if (!managementFlag) {
            managementFlag = true;
            managementToken = r.findManagementKey();
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

        public String findManagementKey() {
            String userDirectory = System.getProperty("user.dir");
            File file = new File(userDirectory + "/key_file.txt");
            try {
                InputStream stream = Files.newInputStream(file.toPath());
                byte[] array = new byte[100];
                stream.read(array);
                String data = new String(array);
                String[] arrOfData = data.split("\n", 2);
                String[] managementArr = arrOfData[0].split(" ", 3);
                return managementArr[2];
            } catch (IOException | ArrayIndexOutOfBoundsException e) {
                return null;
            }
        }

        public boolean keyFile(String managementToken) throws IOException {
            String fileData = " ";
            String userDirectory = System.getProperty("user.dir");
            File file = new File(userDirectory + "/key_file.txt");
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
