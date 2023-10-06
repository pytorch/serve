package org.pytorch.serve.archive.s3;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.file.FileAlreadyExistsException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.archive.utils.ArchiveUtils;
import org.pytorch.serve.archive.utils.InvalidArchiveURLException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Various Http helper routines */
public final class HttpUtils {
    private static final Logger logger = LoggerFactory.getLogger(HttpUtils.class);

    private HttpUtils() {}

    /** Copy model from S3 url to local model store */
    public static boolean copyURLToFile(
            List<String> allowedUrls,
            String url,
            File modelLocation,
            boolean s3SseKmsEnabled,
            String archiveName,
            String store)
            throws FileAlreadyExistsException, IOException, InvalidArchiveURLException {
        if (!ArchiveUtils.validateURL(allowedUrls, url)) {
            return false;
        }

        if (modelLocation.exists()) {
            throw new FileAlreadyExistsException(archiveName);
        }
        // Add if condition to avoid security false alarm
        if (!modelLocation.getPath().toString().startsWith(store)) {
            throw new IOException("Invalid modelLocation:" + modelLocation.getPath().toString());
        }
        // for a simple GET, we have no body so supply the precomputed 'empty' hash
        Map<String, String> headers;
        if (s3SseKmsEnabled) {
            String awsAccessKey = System.getenv("AWS_ACCESS_KEY_ID");
            String awsSecretKey = System.getenv("AWS_SECRET_ACCESS_KEY");
            String regionName = System.getenv("AWS_DEFAULT_REGION");
            if (regionName.isEmpty() || awsAccessKey.isEmpty() || awsSecretKey.isEmpty()) {
                throw new IOException(
                        "Miss environment variables "
                                + "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY or AWS_DEFAULT_REGION");
            }

            headers = new HashMap<>();
            headers.put("x-amz-content-sha256", AWS4SignerBase.EMPTY_BODY_SHA256);

            URL endpointUrl = new URL(url);

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
            HttpURLConnection connection = (HttpURLConnection) endpointUrl.openConnection();
            setHttpConnection(connection, "GET", headers);
            try {
                FileUtils.copyInputStreamToFile(connection.getInputStream(), modelLocation);
            } finally {
                if (connection != null) {
                    connection.disconnect();
                }
            }

        } else {
            URL endpointUrl = new URL(url);
            FileUtils.copyURLToFile(endpointUrl, modelLocation);
        }
        return true;
    }

    public static void setHttpConnection(
            HttpURLConnection connection, String httpMethod, Map<String, String> headers)
            throws IOException {
        connection.setRequestMethod(httpMethod);

        if (headers != null) {
            for (String headerKey : headers.keySet()) {
                connection.setRequestProperty(headerKey, headers.get(headerKey));
            }
        }
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
