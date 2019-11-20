package org.pytorch.serve;

import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import java.security.GeneralSecurityException;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;

public final class TestUtils {

    private TestUtils() {}

    public static void init() {
        // set up system properties for local IDE debug
        if (System.getProperty("tsConfigFile") == null) {
            System.setProperty("tsConfigFile", "src/test/resources/config.properties");
        }
        if (System.getProperty("METRICS_LOCATION") == null) {
            System.setProperty("METRICS_LOCATION", "build/logs");
        }
        if (System.getProperty("LOG_LOCATION") == null) {
            System.setProperty("LOG_LOCATION", "build/logs");
        }

        try {
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(null, InsecureTrustManagerFactory.INSTANCE.getTrustManagers(), null);

            HttpsURLConnection.setDefaultSSLSocketFactory(context.getSocketFactory());

            HttpsURLConnection.setDefaultHostnameVerifier((s, sslSession) -> true);
        } catch (GeneralSecurityException e) {
            // ignore
        }
    }
}
