package org.pytorch.serve.archive.s3;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Various Http helper routines */
public final class HttpUtils {
    private static final Logger logger = LoggerFactory.getLogger(HttpUtils.class);

    private HttpUtils() {}

    /** Copy model from S3 url to local model store */
    public static void copyURLToFile(URL endpointUrl, File modelLocation, boolean s3SseKmsEnabled)
            throws IOException {
        // for a simple GET, we have no body so supply the precomputed 'empty' hash
        Map<String, String> headers;
        if (s3SseKmsEnabled) {
            String awsAccessKey = System.getenv("AWS_ACCESS_KEY_ID");
            String awsSecretKey = System.getenv("AWS_SECRET_ACCESS_KEY");
            String regionName = System.getenv("AWS_DEFAULT_REGION");
            if (!regionName.isEmpty() && !awsAccessKey.isEmpty() && !awsSecretKey.isEmpty()) {
                headers = new HashMap<>();
                headers.put("x-amz-content-sha256", AWS4SignerBase.EMPTY_BODY_SHA256);

                AWS4SignerForAuthorizationHeader signer =
                        new AWS4SignerForAuthorizationHeader(endpointUrl, "GET", "s3", regionName);
                String authorization =
                        signer.computeSignature(
                                headers,
                                null, // no query parameters
                                AWS4SignerBase.EMPTY_BODY_SHA256,
                                awsAccessKey,
                                awsSecretKey);

                // place the computed signature into a formatted 'Authorization' header
                // and call S3
                headers.put("Authorization", authorization);
                HttpURLConnection connection = createHttpConnection(endpointUrl, "GET", headers);
                try {
                    FileUtils.copyInputStreamToFile(connection.getInputStream(), modelLocation);
                } finally {
                    if (connection != null) {
                        connection.disconnect();
                    }
                }
            } else {
                throw new IOException(
                        "Miss environment variables "
                                + "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY or AWS_DEFAULT_REGION");
            }
        } else {
            FileUtils.copyURLToFile(endpointUrl, modelLocation);
        }
    }

    public static HttpURLConnection createHttpConnection(
            URL endpointUrl, String httpMethod, Map<String, String> headers) throws IOException {

        HttpURLConnection connection = (HttpURLConnection) endpointUrl.openConnection();
        connection.setRequestMethod(httpMethod);

        if (headers != null) {
            for (String headerKey : headers.keySet()) {
                connection.setRequestProperty(headerKey, headers.get(headerKey));
            }
        }

        return connection;
    }

    public static String urlEncode(String url, boolean keepPathSlash)
            throws UnsupportedEncodingException {
        String encoded;
        try {
            encoded = URLEncoder.encode(url, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            logger.error("UTF-8 encoding is not supported.", e);
            throw e;
        }
        if (keepPathSlash) {
            encoded = encoded.replace("%2F", "/");
        }
        return encoded;
    }
}
