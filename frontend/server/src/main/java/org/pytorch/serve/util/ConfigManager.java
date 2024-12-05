package org.pytorch.serve.util;

import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.SelfSignedCertificate;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Type;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.security.KeyException;
import java.security.KeyFactory;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collection;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.InvalidPropertiesFormatException;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.io.IOUtils;
import org.pytorch.serve.archive.model.Manifest;
import org.pytorch.serve.device.SystemInfo;
import org.pytorch.serve.metrics.MetricBuilder;
import org.pytorch.serve.servingsdk.snapshot.SnapshotSerializer;
import org.pytorch.serve.snapshot.SnapshotSerializerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ConfigManager {
    // Variables that can be configured through config.properties and Environment Variables
    // NOTE: Variables which can be configured through environment variables **SHOULD** have a
    // "TS_" prefix

    private static final String TS_DEBUG = "debug";
    private static final String TS_INFERENCE_ADDRESS = "inference_address";
    private static final String TS_MANAGEMENT_ADDRESS = "management_address";
    private static final String TS_METRICS_ADDRESS = "metrics_address";
    private static final String TS_LOAD_MODELS = "load_models";
    private static final String TS_BLACKLIST_ENV_VARS = "blacklist_env_vars";
    private static final String TS_DEFAULT_WORKERS_PER_MODEL = "default_workers_per_model";
    private static final String TS_DEFAULT_RESPONSE_TIMEOUT = "default_response_timeout";
    private static final String TS_DEFAULT_STARTUP_TIMEOUT = "default_startup_timeout";
    private static final String TS_UNREGISTER_MODEL_TIMEOUT = "unregister_model_timeout";
    private static final String TS_NUMBER_OF_NETTY_THREADS = "number_of_netty_threads";
    private static final String TS_NETTY_CLIENT_THREADS = "netty_client_threads";
    private static final String TS_JOB_QUEUE_SIZE = "job_queue_size";
    private static final String TS_NUMBER_OF_GPU = "number_of_gpu";
    private static final String TS_METRICS_CONFIG = "metrics_config";
    private static final String TS_METRICS_MODE = "metrics_mode";
    private static final String TS_MODEL_METRICS_AUTO_DETECT = "model_metrics_auto_detect";
    private static final String TS_DISABLE_SYSTEM_METRICS = "disable_system_metrics";

    // IPEX config option that can be set at config.properties
    private static final String TS_IPEX_ENABLE = "ipex_enable";
    private static final String TS_CPU_LAUNCHER_ENABLE = "cpu_launcher_enable";
    private static final String TS_CPU_LAUNCHER_ARGS = "cpu_launcher_args";
    private static final String TS_IPEX_GPU_ENABLE = "ipex_gpu_enable";

    private static final String TS_ASYNC_LOGGING = "async_logging";
    private static final String TS_CORS_ALLOWED_ORIGIN = "cors_allowed_origin";
    private static final String TS_CORS_ALLOWED_METHODS = "cors_allowed_methods";
    private static final String TS_CORS_ALLOWED_HEADERS = "cors_allowed_headers";
    private static final String TS_DECODE_INPUT_REQUEST = "decode_input_request";
    private static final String TS_KEYSTORE = "keystore";
    private static final String TS_KEYSTORE_PASS = "keystore_pass";
    private static final String TS_KEYSTORE_TYPE = "keystore_type";
    private static final String TS_CERTIFICATE_FILE = "certificate_file";
    private static final String TS_PRIVATE_KEY_FILE = "private_key_file";
    private static final String TS_MAX_REQUEST_SIZE = "max_request_size";
    private static final String TS_MAX_RESPONSE_SIZE = "max_response_size";
    private static final String TS_LIMIT_MAX_IMAGE_PIXELS = "limit_max_image_pixels";
    private static final String TS_DEFAULT_SERVICE_HANDLER = "default_service_handler";
    private static final String TS_SERVICE_ENVELOPE = "service_envelope";
    private static final String TS_MODEL_SERVER_HOME = "model_server_home";
    private static final String TS_MODEL_STORE = "model_store";
    private static final String TS_PREFER_DIRECT_BUFFER = "prefer_direct_buffer";
    private static final String TS_ALLOWED_URLS = "allowed_urls";
    private static final String TS_INSTALL_PY_DEP_PER_MODEL = "install_py_dep_per_model";
    private static final String TS_ENABLE_METRICS_API = "enable_metrics_api";
    private static final String TS_GRPC_INFERENCE_ADDRESS = "grpc_inference_address";
    private static final String TS_GRPC_MANAGEMENT_ADDRESS = "grpc_management_address";
    private static final String TS_GRPC_INFERENCE_PORT = "grpc_inference_port";
    private static final String TS_GRPC_MANAGEMENT_PORT = "grpc_management_port";
    private static final String TS_GRPC_INFERENCE_MAX_CONNECTION_AGE_MS =
            "grpc_inference_max_connection_age_ms";
    private static final String TS_GRPC_MANAGEMENT_MAX_CONNECTION_AGE_MS =
            "grpc_management_max_connection_age_ms";
    private static final String TS_GRPC_INFERENCE_MAX_CONNECTION_AGE_GRACE_MS =
            "grpc_inference_max_connection_age_grace_ms";
    private static final String TS_GRPC_MANAGEMENT_MAX_CONNECTION_AGE_GRACE_MS =
            "grpc_management_max_connection_age_grace_ms";
    private static final String TS_ENABLE_GRPC_SSL = "enable_grpc_ssl";
    private static final String TS_INITIAL_WORKER_PORT = "initial_worker_port";
    private static final String TS_INITIAL_DISTRIBUTION_PORT = "initial_distribution_port";
    private static final String TS_WORKFLOW_STORE = "workflow_store";
    private static final String TS_CPP_LOG_CONFIG = "cpp_log_config";
    private static final String TS_OPEN_INFERENCE_PROTOCOL = "ts_open_inference_protocol";
    private static final String TS_TOKEN_EXPIRATION_TIME_MIN = "token_expiration_min";
    private static final String TS_HEADER_KEY_SEQUENCE_ID = "ts_header_key_sequence_id";
    private static final String TS_HEADER_KEY_SEQUENCE_START = "ts_header_key_sequence_start";
    private static final String TS_HEADER_KEY_SEQUENCE_END = "ts_header_key_sequence_end";
    private static final String TS_DISABLE_TOKEN_AUTHORIZATION = "disable_token_authorization";
    private static final String TS_ENABLE_MODEL_API = "enable_model_api";

    // Configuration which are not documented or enabled through environment
    // variables
    private static final String USE_NATIVE_IO = "use_native_io";
    private static final String IO_RATIO = "io_ratio";
    private static final String METRIC_TIME_INTERVAL = "metric_time_interval";
    private static final String ENABLE_ENVVARS_CONFIG = "enable_envvars_config";
    private static final String MODEL_SNAPSHOT = "model_snapshot";
    private static final String MODEL_CONFIG = "models";
    private static final String VERSION = "version";
    private static final String SYSTEM_METRICS_CMD = "system_metrics_cmd";

    // Configuration default values
    private static final String DEFAULT_TS_ALLOWED_URLS = "file://.*|http(s)?://.*";
    private static final String USE_ENV_ALLOWED_URLS = "use_env_allowed_urls";

    // Variables which are local
    public static final String MODEL_METRICS_LOGGER = "MODEL_METRICS";
    public static final String MODEL_LOGGER = "MODEL_LOG";
    public static final String MODEL_SERVER_METRICS_LOGGER = "TS_METRICS";
    public static final String MODEL_SERVER_TELEMETRY_LOGGER = "TELEMETRY_METRICS";

    public static final String METRIC_FORMAT_PROMETHEUS = "prometheus";

    public static final String PYTHON_EXECUTABLE = "python";

    public static final String DEFAULT_REQUEST_SEQUENCE_ID = "ts_request_sequence_id";
    public static final String DEFAULT_REQUEST_SEQUENCE_START = "ts_request_sequence_start";
    public static final String DEFAULT_REQUEST_SEQUENCE_END = "ts_request_sequence_end";

    public static final Pattern ADDRESS_PATTERN =
            Pattern.compile(
                    "((https|http)://([^:^/]+)(:([0-9]+))?)|(unix:(/.*))",
                    Pattern.CASE_INSENSITIVE);
    private static Pattern pattern = Pattern.compile("\\$\\$([^$]+[^$])\\$\\$");

    private Pattern blacklistPattern;
    private Properties prop;

    private boolean snapshotDisabled;

    private static ConfigManager instance;
    private String hostName;
    private Map<String, Map<String, JsonObject>> modelConfig = new HashMap<>();
    private String torchrunLogDir;
    private boolean telemetryEnabled;
    private String headerKeySequenceId;
    private String headerKeySequenceStart;
    private String headerKeySequenceEnd;

    public SystemInfo systemInfo;

    private static final Logger logger = LoggerFactory.getLogger(ConfigManager.class);

    private ConfigManager(Arguments args) throws IOException {
        prop = new Properties();
        this.systemInfo = new SystemInfo();

        this.snapshotDisabled = args.isSnapshotDisabled();
        String version = readFile(getModelServerHome() + "/ts/version.txt");
        if (version != null) {
            version = version.replaceAll("[\\n\\t ]", "");
            prop.setProperty(VERSION, version);
        }

        String logLocation = System.getenv("LOG_LOCATION");
        if (logLocation != null) {
            System.setProperty("LOG_LOCATION", logLocation);
        } else if (System.getProperty("LOG_LOCATION") == null) {
            System.setProperty("LOG_LOCATION", "logs");
        }

        String metricsLocation = System.getenv("METRICS_LOCATION");
        if (metricsLocation != null) {
            System.setProperty("METRICS_LOCATION", metricsLocation);
        } else if (System.getProperty("METRICS_LOCATION") == null) {
            System.setProperty("METRICS_LOCATION", "logs");
        }

        String filePath = System.getenv("TS_CONFIG_FILE");
        Properties snapshotConfig = null;

        if (filePath == null) {
            filePath = args.getTsConfigFile();
            if (filePath == null) {
                snapshotConfig = getLastSnapshot();
                if (snapshotConfig == null) {
                    filePath = System.getProperty("tsConfigFile", "config.properties");
                } else {
                    prop.putAll(snapshotConfig);
                }
            }
        }

        if (filePath != null) {
            File tsConfigFile = new File(filePath);
            if (tsConfigFile.exists()) {
                try (InputStream stream = Files.newInputStream(tsConfigFile.toPath())) {
                    prop.load(stream);
                    prop.put("tsConfigFile", filePath);
                } catch (IOException e) {
                    throw new IllegalStateException("Unable to read configuration file", e);
                }
            }
        }

        if (System.getenv("SM_TELEMETRY_LOG") != null) {
            telemetryEnabled = true;
        } else {
            telemetryEnabled = false;
        }
        resolveEnvVarVals(prop);

        String modelStore = args.getModelStore();
        if (modelStore != null) {
            prop.setProperty(TS_MODEL_STORE, modelStore);
        }

        String workflowStore = args.getWorkflowStore();
        if (workflowStore != null) {
            prop.setProperty(TS_WORKFLOW_STORE, workflowStore);
        } else if (prop.getProperty(TS_WORKFLOW_STORE) == null) {
            prop.setProperty(TS_WORKFLOW_STORE, prop.getProperty(TS_MODEL_STORE));
        }

        String cppLogConfigFile = args.getCppLogConfigFile();
        if (cppLogConfigFile != null) {
            prop.setProperty(TS_CPP_LOG_CONFIG, cppLogConfigFile);
        }

        String[] models = args.getModels();
        if (models != null) {
            prop.setProperty(TS_LOAD_MODELS, String.join(",", models));
        }

        if (args.isModelEnabled().equals("true")) {
            prop.setProperty(TS_ENABLE_MODEL_API, args.isModelEnabled());
        }

        String tokenDisabled = args.isTokenDisabled();
        if (tokenDisabled.equals("true")) {
            prop.setProperty(TS_DISABLE_TOKEN_AUTHORIZATION, tokenDisabled);
        }

        prop.setProperty(
                TS_NUMBER_OF_GPU,
                String.valueOf(
                        Integer.min(
                                this.systemInfo.getNumberOfAccelerators(),
                                getIntProperty(TS_NUMBER_OF_GPU, Integer.MAX_VALUE))));

        String pythonExecutable = args.getPythonExecutable();
        if (pythonExecutable != null) {
            prop.setProperty(PYTHON_EXECUTABLE, pythonExecutable);
        }

        try {
            InetAddress ip = InetAddress.getLocalHost();
            hostName = ip.getHostName();
        } catch (UnknownHostException e) {
            hostName = "Unknown";
        }

        if (Boolean.parseBoolean(prop.getProperty(TS_ASYNC_LOGGING))) {
            enableAsyncLogging();
        }

        if (Boolean.parseBoolean(getEnableEnvVarsConfig())) {
            // Environment variables have higher precedence over the config file variables
            setSystemVars();
        }

        setModelConfig();
        setTsHeaderKeySequenceId();
        setTsHeaderKeySequenceStart();
        setTsHeaderKeySequenceEnd();

        // Issue warnining about URLs that can be accessed when loading models
        if (prop.getProperty(TS_ALLOWED_URLS, DEFAULT_TS_ALLOWED_URLS) == DEFAULT_TS_ALLOWED_URLS) {
            logger.warn(
                    "Your torchserve instance can access any URL to load models. "
                            + "When deploying to production, make sure to limit the set of allowed_urls in config.properties");
        }
    }

    public static String readFile(String path) throws IOException {
        return Files.readString(Paths.get(path));
    }

    private void resolveEnvVarVals(Properties prop) {
        Set<String> keys = prop.stringPropertyNames();
        for (String key : keys) {
            String val = prop.getProperty(key);
            Matcher matcher = pattern.matcher(val);
            if (matcher.find()) {
                StringBuffer sb = new StringBuffer();
                do {
                    String envVar = matcher.group(1);
                    if (System.getenv(envVar) == null) {
                        throw new IllegalArgumentException(
                                "Invalid Environment Variable " + envVar);
                    }
                    matcher.appendReplacement(sb, System.getenv(envVar));
                } while (matcher.find());
                matcher.appendTail(sb);
                prop.setProperty(key, sb.toString());
            }
        }
    }

    private void setSystemVars() {
        Class<ConfigManager> configClass = ConfigManager.class;
        Field[] fields = configClass.getDeclaredFields();
        for (Field f : fields) {
            // For security, disable TS_ALLOWED_URLS in env.
            if ("TS_ALLOWED_URLS".equals(f.getName())
                    && !"true"
                            .equals(
                                    prop.getProperty(USE_ENV_ALLOWED_URLS, "false")
                                            .toLowerCase())) {
                continue;
            }
            if (f.getName().startsWith("TS_")) {
                String val = System.getenv(f.getName());
                if (val != null) {
                    try {
                        prop.setProperty((String) f.get(ConfigManager.class), val);
                    } catch (IllegalAccessException e) {
                        e.printStackTrace(); // NOPMD
                    }
                }
            }
        }
    }

    public String getEnableEnvVarsConfig() {
        return prop.getProperty(ENABLE_ENVVARS_CONFIG, "false");
    }

    public String getHostName() {
        return hostName;
    }

    public static void init(Arguments args) throws IOException {
        instance = new ConfigManager(args);
    }

    public static ConfigManager getInstance() {
        return instance;
    }

    public boolean isDebug() {
        return Boolean.getBoolean("TS_DEBUG")
                || Boolean.parseBoolean(prop.getProperty(TS_DEBUG, "false"));
    }

    public Connector getListener(ConnectorType connectorType) {
        String binding;
        switch (connectorType) {
            case MANAGEMENT_CONNECTOR:
                binding = prop.getProperty(TS_MANAGEMENT_ADDRESS, "http://127.0.0.1:8081");
                break;
            case METRICS_CONNECTOR:
                binding = prop.getProperty(TS_METRICS_ADDRESS, "http://127.0.0.1:8082");
                break;
            default:
                binding = prop.getProperty(TS_INFERENCE_ADDRESS, "http://127.0.0.1:8080");
        }
        return Connector.parse(binding, connectorType);
    }

    public InetAddress getGRPCAddress(ConnectorType connectorType)
            throws UnknownHostException, IllegalArgumentException {
        if (connectorType == ConnectorType.MANAGEMENT_CONNECTOR) {
            return InetAddress.getByName(prop.getProperty(TS_GRPC_MANAGEMENT_ADDRESS, "127.0.0.1"));
        } else if (connectorType == ConnectorType.INFERENCE_CONNECTOR) {
            return InetAddress.getByName(prop.getProperty(TS_GRPC_INFERENCE_ADDRESS, "127.0.0.1"));
        } else {
            throw new IllegalArgumentException(
                    "Connector type not supported by gRPC: " + connectorType);
        }
    }

    public int getGRPCPort(ConnectorType connectorType) throws IllegalArgumentException {
        String port;
        if (connectorType == ConnectorType.MANAGEMENT_CONNECTOR) {
            port = prop.getProperty(TS_GRPC_MANAGEMENT_PORT, "7071");
        } else if (connectorType == ConnectorType.INFERENCE_CONNECTOR) {
            port = prop.getProperty(TS_GRPC_INFERENCE_PORT, "7070");
        } else {
            throw new IllegalArgumentException(
                    "Connector type not supported by gRPC: " + connectorType);
        }
        return Integer.parseInt(port);
    }

    public long getGRPCMaxConnectionAge(ConnectorType connectorType)
            throws IllegalArgumentException {
        if (connectorType == ConnectorType.MANAGEMENT_CONNECTOR) {
            return getLongProperty(TS_GRPC_MANAGEMENT_MAX_CONNECTION_AGE_MS, Long.MAX_VALUE);
        } else if (connectorType == ConnectorType.INFERENCE_CONNECTOR) {
            return getLongProperty(TS_GRPC_INFERENCE_MAX_CONNECTION_AGE_MS, Long.MAX_VALUE);
        } else {
            throw new IllegalArgumentException(
                    "Connector type not supported by gRPC: " + connectorType);
        }
    }

    public long getGRPCMaxConnectionAgeGrace(ConnectorType connectorType)
            throws IllegalArgumentException {
        if (connectorType == ConnectorType.MANAGEMENT_CONNECTOR) {
            return getLongProperty(TS_GRPC_MANAGEMENT_MAX_CONNECTION_AGE_GRACE_MS, Long.MAX_VALUE);
        } else if (connectorType == ConnectorType.INFERENCE_CONNECTOR) {
            return getLongProperty(TS_GRPC_INFERENCE_MAX_CONNECTION_AGE_GRACE_MS, Long.MAX_VALUE);
        } else {
            throw new IllegalArgumentException(
                    "Connector type not supported by gRPC: " + connectorType);
        }
    }

    public boolean isOpenInferenceProtocol() {
        String inferenceProtocol = System.getenv("TS_OPEN_INFERENCE_PROTOCOL");
        if (inferenceProtocol != null && inferenceProtocol != "") {
            return "oip".equals(inferenceProtocol);
        }
        return Boolean.parseBoolean(prop.getProperty(TS_OPEN_INFERENCE_PROTOCOL, "false"));
    }

    public boolean isGRPCSSLEnabled() {
        return Boolean.parseBoolean(getProperty(TS_ENABLE_GRPC_SSL, "false"));
    }

    public boolean getPreferDirectBuffer() {
        return Boolean.parseBoolean(getProperty(TS_PREFER_DIRECT_BUFFER, "false"));
    }

    public boolean getInstallPyDepPerModel() {
        return Boolean.parseBoolean(getProperty(TS_INSTALL_PY_DEP_PER_MODEL, "false"));
    }

    public boolean isMetricApiEnable() {
        return Boolean.parseBoolean(getProperty(TS_ENABLE_METRICS_API, "true"));
    }

    public boolean isCPULauncherEnabled() {
        return Boolean.parseBoolean(getProperty(TS_CPU_LAUNCHER_ENABLE, "false"));
    }

    public String getCPULauncherArgs() {
        return getProperty(TS_CPU_LAUNCHER_ARGS, null);
    }

    public boolean isIPEXGpuEnabled() {
        return Boolean.parseBoolean(getProperty(TS_IPEX_GPU_ENABLE, "false"));
    }

    public boolean getDisableTokenAuthorization() {
        return Boolean.parseBoolean(getProperty(TS_DISABLE_TOKEN_AUTHORIZATION, "false"));
    }

    public int getNettyThreads() {
        return getIntProperty(TS_NUMBER_OF_NETTY_THREADS, 0);
    }

    public int getNettyClientThreads() {
        return getIntProperty(TS_NETTY_CLIENT_THREADS, 0);
    }

    public int getJobQueueSize() {
        return getIntProperty(TS_JOB_QUEUE_SIZE, 100);
    }

    public int getNumberOfGpu() {
        return getIntProperty(TS_NUMBER_OF_GPU, 0);
    }

    public boolean isModelApiEnabled() {
        return Boolean.parseBoolean(getProperty(TS_ENABLE_MODEL_API, "false"));
    }

    public String getMetricsConfigPath() {
        String path = getCanonicalPath(prop.getProperty(TS_METRICS_CONFIG));
        if (path == null) {
            path = getModelServerHome() + "/ts/configs/metrics.yaml";
        }
        return path;
    }

    public String getTorchRunLogDir() {
        if (torchrunLogDir == null) {
            torchrunLogDir =
                    Paths.get(
                                    getCanonicalPath(System.getProperty("LOG_LOCATION")),
                                    "torchelastic_ts")
                            .toString();
        }
        return torchrunLogDir;
    }

    public MetricBuilder.MetricMode getMetricsMode() {
        String metricsMode = getProperty(TS_METRICS_MODE, "log");
        try {
            return MetricBuilder.MetricMode.valueOf(
                    metricsMode.replaceAll("\\s", "").toUpperCase());
        } catch (IllegalArgumentException | NullPointerException e) {
            logger.error(
                    "Configured metrics mode \"{}\" not supported. Defaulting to \"{}\" mode: {}",
                    metricsMode,
                    MetricBuilder.MetricMode.LOG,
                    e);
            return MetricBuilder.MetricMode.LOG;
        }
    }

    public boolean isModelMetricsAutoDetectEnabled() {
        return Boolean.parseBoolean(getProperty(TS_MODEL_METRICS_AUTO_DETECT, "false"));
    }

    public boolean isSystemMetricsDisabled() {
        return Boolean.parseBoolean(getProperty(TS_DISABLE_SYSTEM_METRICS, "false"));
    }

    public String getTsDefaultServiceHandler() {
        return getProperty(TS_DEFAULT_SERVICE_HANDLER, null);
    }

    public String getTsServiceEnvelope() {
        return getProperty(TS_SERVICE_ENVELOPE, null);
    }

    public Properties getConfiguration() {
        return (Properties) prop.clone();
    }

    public int getConfiguredDefaultWorkersPerModel() {
        return getIntProperty(TS_DEFAULT_WORKERS_PER_MODEL, 0);
    }

    public int getDefaultWorkers() {
        if (isDebug()) {
            return 1;
        }
        int workers = getConfiguredDefaultWorkersPerModel();

        if (workers == 0) {
            workers = getNumberOfGpu();
        }
        if (workers == 0) {
            workers = Runtime.getRuntime().availableProcessors();
        }

        return workers;
    }

    public int getMetricTimeInterval() {
        return getIntProperty(METRIC_TIME_INTERVAL, 60);
    }

    public String getModelServerHome() {
        String tsHome = System.getenv("TS_MODEL_SERVER_HOME");
        if (tsHome == null) {
            tsHome = System.getProperty(TS_MODEL_SERVER_HOME);
            if (tsHome == null) {
                tsHome = getProperty(TS_MODEL_SERVER_HOME, null);
                if (tsHome == null) {
                    tsHome = getCanonicalPath(findTsHome());
                    return tsHome;
                }
            }
        }

        File dir = new File(tsHome);
        if (!dir.exists()) {
            throw new IllegalArgumentException("Model server home not exist: " + tsHome);
        }
        tsHome = getCanonicalPath(dir);
        return tsHome;
    }

    public String getPythonExecutable() {
        return prop.getProperty(PYTHON_EXECUTABLE, "python");
    }

    public String getModelStore() {
        return getCanonicalPath(prop.getProperty(TS_MODEL_STORE));
    }

    public String getWorkflowStore() {
        return getCanonicalPath(prop.getProperty(TS_WORKFLOW_STORE));
    }

    public String getTsCppLogConfig() {
        return prop.getProperty(TS_CPP_LOG_CONFIG, null);
    }

    public String getModelSnapshot() {
        return prop.getProperty(MODEL_SNAPSHOT, null);
    }

    public String getLoadModels() {
        return prop.getProperty(TS_LOAD_MODELS);
    }

    public Pattern getBlacklistPattern() {
        return blacklistPattern;
    }

    public String getCorsAllowedOrigin() {
        return prop.getProperty(TS_CORS_ALLOWED_ORIGIN);
    }

    public String getCorsAllowedMethods() {
        return prop.getProperty(TS_CORS_ALLOWED_METHODS);
    }

    public String getCorsAllowedHeaders() {
        return prop.getProperty(TS_CORS_ALLOWED_HEADERS);
    }

    public String getPrivateKeyFile() {
        return prop.getProperty(TS_PRIVATE_KEY_FILE);
    }

    public String getCertificateFile() {
        return prop.getProperty(TS_CERTIFICATE_FILE);
    }

    public String getSystemMetricsCmd() {
        return prop.getProperty(SYSTEM_METRICS_CMD, "");
    }

    public SslContext getSslContext() throws IOException, GeneralSecurityException {
        List<String> supportedCiphers =
                Arrays.asList(
                        "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
                        "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256");

        PrivateKey privateKey;
        X509Certificate[] chain;
        String keyStoreFile = prop.getProperty(TS_KEYSTORE);
        String privateKeyFile = prop.getProperty(TS_PRIVATE_KEY_FILE);
        String certificateFile = prop.getProperty(TS_CERTIFICATE_FILE);
        if (keyStoreFile != null) {
            char[] keystorePass = getProperty(TS_KEYSTORE_PASS, "changeit").toCharArray();
            String keystoreType = getProperty(TS_KEYSTORE_TYPE, "PKCS12");
            KeyStore keyStore = KeyStore.getInstance(keystoreType);
            try (InputStream is = Files.newInputStream(Paths.get(keyStoreFile))) {
                keyStore.load(is, keystorePass);
            }

            Enumeration<String> en = keyStore.aliases();
            String keyAlias = null;
            while (en.hasMoreElements()) {
                String alias = en.nextElement();
                if (keyStore.isKeyEntry(alias)) {
                    keyAlias = alias;
                    break;
                }
            }

            if (keyAlias == null) {
                throw new KeyException("No key entry found in keystore.");
            }

            privateKey = (PrivateKey) keyStore.getKey(keyAlias, keystorePass);

            Certificate[] certs = keyStore.getCertificateChain(keyAlias);
            chain = new X509Certificate[certs.length];
            for (int i = 0; i < certs.length; ++i) {
                chain[i] = (X509Certificate) certs[i];
            }
        } else if (privateKeyFile != null && certificateFile != null) {
            privateKey = loadPrivateKey(privateKeyFile);
            chain = loadCertificateChain(certificateFile);
        } else {
            SelfSignedCertificate ssc = new SelfSignedCertificate();
            privateKey = ssc.key();
            chain = new X509Certificate[] {ssc.cert()};
        }

        return SslContextBuilder.forServer(privateKey, chain)
                .protocols(new String[] {"TLSv1.2"})
                .ciphers(supportedCiphers)
                .build();
    }

    private PrivateKey loadPrivateKey(String keyFile) throws IOException, GeneralSecurityException {
        KeyFactory keyFactory = KeyFactory.getInstance("RSA");
        try (InputStream is = Files.newInputStream(Paths.get(keyFile))) {
            String content = IOUtils.toString(is, StandardCharsets.UTF_8);
            content = content.replaceAll("-----(BEGIN|END)( RSA)? PRIVATE KEY-----\\s*", "");
            byte[] buf = Base64.getMimeDecoder().decode(content);
            try {
                PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(buf);
                return keyFactory.generatePrivate(privKeySpec);
            } catch (InvalidKeySpecException e) {
                // old private key is OpenSSL format private key
                buf = OpenSslKey.convertPrivateKey(buf);
                PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(buf);
                return keyFactory.generatePrivate(privKeySpec);
            }
        }
    }

    private X509Certificate[] loadCertificateChain(String keyFile)
            throws IOException, GeneralSecurityException {
        CertificateFactory cf = CertificateFactory.getInstance("X.509");
        try (InputStream is = Files.newInputStream(Paths.get(keyFile))) {
            Collection<? extends Certificate> certs = cf.generateCertificates(is);
            int i = 0;
            X509Certificate[] chain = new X509Certificate[certs.size()];
            for (Certificate cert : certs) {
                chain[i++] = (X509Certificate) cert;
            }
            return chain;
        }
    }

    private Properties getLastSnapshot() {
        if (isSnapshotDisabled()) {
            return null;
        }
        SnapshotSerializer serializer = SnapshotSerializerFactory.getSerializer();
        return serializer.getLastSnapshot();
    }

    public String getProperty(String key, String def) {
        return prop.getProperty(key, def);
    }

    public void validateConfigurations() throws InvalidPropertiesFormatException {
        String blacklistVars = prop.getProperty(TS_BLACKLIST_ENV_VARS, "");
        try {
            blacklistPattern = Pattern.compile(blacklistVars);
        } catch (PatternSyntaxException e) {
            throw new InvalidPropertiesFormatException(e);
        }
    }

    public String dumpConfigurations() {
        Runtime runtime = Runtime.getRuntime();
        return "\nTorchserve version: "
                + prop.getProperty(VERSION)
                + "\nTS Home: "
                + getModelServerHome()
                + "\nCurrent directory: "
                + getCanonicalPath(".")
                + "\nTemp directory: "
                + System.getProperty("java.io.tmpdir")
                + "\nMetrics config path: "
                + getMetricsConfigPath()
                + "\nNumber of GPUs: "
                + getNumberOfGpu()
                + "\nNumber of CPUs: "
                + runtime.availableProcessors()
                + "\nMax heap size: "
                + (runtime.maxMemory() / 1024 / 1024)
                + " M\nPython executable: "
                + (getPythonExecutable() == null ? "N/A" : getPythonExecutable())
                + "\nConfig file: "
                + prop.getProperty("tsConfigFile", "N/A")
                + "\nInference address: "
                + getListener(ConnectorType.INFERENCE_CONNECTOR)
                + "\nManagement address: "
                + getListener(ConnectorType.MANAGEMENT_CONNECTOR)
                + "\nMetrics address: "
                + getListener(ConnectorType.METRICS_CONNECTOR)
                + "\nModel Store: "
                + (getModelStore() == null ? "N/A" : getModelStore())
                + "\nInitial Models: "
                + (getLoadModels() == null ? "N/A" : getLoadModels())
                + "\nLog dir: "
                + getCanonicalPath(System.getProperty("LOG_LOCATION"))
                + "\nMetrics dir: "
                + getCanonicalPath(System.getProperty("METRICS_LOCATION"))
                + "\nNetty threads: "
                + getNettyThreads()
                + "\nNetty client threads: "
                + getNettyClientThreads()
                + "\nDefault workers per model: "
                + getDefaultWorkers()
                + "\nBlacklist Regex: "
                + prop.getProperty(TS_BLACKLIST_ENV_VARS, "N/A")
                + "\nMaximum Response Size: "
                + prop.getProperty(TS_MAX_RESPONSE_SIZE, "6553500")
                + "\nMaximum Request Size: "
                + prop.getProperty(TS_MAX_REQUEST_SIZE, "6553500")
                + "\nLimit Maximum Image Pixels: "
                + prop.getProperty(TS_LIMIT_MAX_IMAGE_PIXELS, "true")
                + "\nPrefer direct buffer: "
                + prop.getProperty(TS_PREFER_DIRECT_BUFFER, "false")
                + "\nAllowed Urls: "
                + getAllowedUrls()
                + "\nCustom python dependency for model allowed: "
                + prop.getProperty(TS_INSTALL_PY_DEP_PER_MODEL, "false")
                + "\nEnable metrics API: "
                + prop.getProperty(TS_ENABLE_METRICS_API, "true")
                + "\nMetrics mode: "
                + getMetricsMode()
                + "\nDisable system metrics: "
                + isSystemMetricsDisabled()
                + "\nWorkflow Store: "
                + (getWorkflowStore() == null ? "N/A" : getWorkflowStore())
                + "\nCPP log config: "
                + (getTsCppLogConfig() == null ? "N/A" : getTsCppLogConfig())
                + "\nModel config: "
                + prop.getProperty(MODEL_CONFIG, "N/A")
                + "\nSystem metrics command: "
                + (getSystemMetricsCmd().isEmpty() ? "default" : getSystemMetricsCmd())
                + "\nModel API enabled: "
                + (isModelApiEnabled() ? "true" : "false");
    }

    public boolean useNativeIo() {
        return Boolean.parseBoolean(prop.getProperty(USE_NATIVE_IO, "true"));
    }

    public int getIoRatio() {
        return getIntProperty(IO_RATIO, 50);
    }

    public int getMaxResponseSize() {
        return getIntProperty(TS_MAX_RESPONSE_SIZE, 6553500);
    }

    public int getMaxRequestSize() {
        return getIntProperty(TS_MAX_REQUEST_SIZE, 6553500);
    }

    public boolean isLimitMaxImagePixels() {
        return Boolean.parseBoolean(prop.getProperty(TS_LIMIT_MAX_IMAGE_PIXELS, "true"));
    }

    public void setProperty(String key, String value) {
        prop.setProperty(key, value);
    }

    private int getIntProperty(String key, int def) {
        String value = prop.getProperty(key);
        if (value == null) {
            return def;
        }
        return Integer.parseInt(value);
    }

    private long getLongProperty(String key, long def) {
        String value = prop.getProperty(key);
        if (value == null) {
            return def;
        }
        return Long.parseLong(value);
    }

    public int getDefaultResponseTimeout() {
        return Integer.parseInt(prop.getProperty(TS_DEFAULT_RESPONSE_TIMEOUT, "120"));
    }

    public int getDefaultStartupTimeout() {
        return Integer.parseInt(prop.getProperty(TS_DEFAULT_STARTUP_TIMEOUT, "120"));
    }

    public int getUnregisterModelTimeout() {
        return Integer.parseInt(prop.getProperty(TS_UNREGISTER_MODEL_TIMEOUT, "120"));
    }

    private File findTsHome() {
        File cwd = new File(getCanonicalPath("."));
        File file = cwd;
        while (file != null) {
            File ts = new File(file, "ts");
            if (ts.exists()) {
                return file;
            }
            file = file.getParentFile();
        }
        return cwd;
    }

    private void enableAsyncLogging() {
        System.setProperty(
                "log4j2.contextSelector",
                "org.apache.logging.log4j.core.async.AsyncLoggerContextSelector");
    }

    public HashMap<String, String> getBackendConfiguration() {
        HashMap<String, String> config = new HashMap<>();
        // Append properties used by backend worker here
        config.put("TS_DECODE_INPUT_REQUEST", prop.getProperty(TS_DECODE_INPUT_REQUEST, "true"));
        config.put("TS_IPEX_ENABLE", prop.getProperty(TS_IPEX_ENABLE, "false"));
        config.put("TS_IPEX_GPU_ENABLE", prop.getProperty(TS_IPEX_GPU_ENABLE, "false"));
        return config;
    }

    private static String getCanonicalPath(File file) {
        try {
            return file.getCanonicalPath();
        } catch (IOException e) {
            return file.getAbsolutePath();
        }
    }

    private static String getCanonicalPath(String path) {
        if (path == null) {
            return null;
        }
        return getCanonicalPath(new File(path));
    }

    public List<String> getAllowedUrls() {
        String allowedURL = prop.getProperty(TS_ALLOWED_URLS, DEFAULT_TS_ALLOWED_URLS);
        return Arrays.asList(allowedURL.split(","));
    }

    public boolean isSnapshotDisabled() {
        return snapshotDisabled;
    }

    public Double getTimeToExpiration() {
        if (prop.getProperty(TS_TOKEN_EXPIRATION_TIME_MIN) != null) {
            try {
                return Double.valueOf(prop.getProperty(TS_TOKEN_EXPIRATION_TIME_MIN));
            } catch (NumberFormatException e) {
                logger.error("Token expiration not a valid integer");
            }
        }
        return 60.0;
    }

    public String getTsHeaderKeySequenceId() {
        return this.headerKeySequenceId;
    }

    public void setTsHeaderKeySequenceId() {
        this.headerKeySequenceId =
                prop.getProperty(TS_HEADER_KEY_SEQUENCE_ID, DEFAULT_REQUEST_SEQUENCE_ID);
    }

    public String getTsHeaderKeySequenceStart() {
        return this.headerKeySequenceStart;
    }

    public void setTsHeaderKeySequenceStart() {
        this.headerKeySequenceStart =
                prop.getProperty(TS_HEADER_KEY_SEQUENCE_START, DEFAULT_REQUEST_SEQUENCE_START);
    }

    public String getTsHeaderKeySequenceEnd() {
        return this.headerKeySequenceEnd;
    }

    public void setTsHeaderKeySequenceEnd() {
        this.headerKeySequenceEnd =
                prop.getProperty(TS_HEADER_KEY_SEQUENCE_END, DEFAULT_REQUEST_SEQUENCE_END);
    }

    public boolean isSSLEnabled(ConnectorType connectorType) {
        String address = prop.getProperty(TS_INFERENCE_ADDRESS, "http://127.0.0.1:8080");
        switch (connectorType) {
            case MANAGEMENT_CONNECTOR:
                address = prop.getProperty(TS_MANAGEMENT_ADDRESS, "http://127.0.0.1:8081");
                break;
            case METRICS_CONNECTOR:
                address = prop.getProperty(TS_METRICS_ADDRESS, "http://127.0.0.1:8082");
                break;
            default:
                break;
        }
        // String inferenceAddress = prop.getProperty(TS_INFERENCE_ADDRESS,
        // "http://127.0.0.1:8080");
        Matcher matcher = ConfigManager.ADDRESS_PATTERN.matcher(address);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Invalid binding address: " + address);
        }

        String protocol = matcher.group(2);

        return "https".equalsIgnoreCase(protocol);
    }

    public int getInitialWorkerPort() {
        return Integer.parseInt(prop.getProperty(TS_INITIAL_WORKER_PORT, "9000"));
    }

    public void setInitialWorkerPort(int initialPort) {
        prop.setProperty(TS_INITIAL_WORKER_PORT, String.valueOf(initialPort));
    }

    public int getInitialDistributionPort() {
        return Integer.parseInt(prop.getProperty(TS_INITIAL_DISTRIBUTION_PORT, "29500"));
    }

    public void setInitialDistributionPort(int initialPort) {
        prop.setProperty(TS_INITIAL_DISTRIBUTION_PORT, String.valueOf(initialPort));
    }

    private void setModelConfig() {
        String modelConfigStr = prop.getProperty(MODEL_CONFIG, null);
        Type type = new TypeToken<Map<String, Map<String, JsonObject>>>() {}.getType();

        if (modelConfigStr != null) {
            this.modelConfig = JsonUtils.GSON.fromJson(modelConfigStr, type);
        }
    }

    public int getJsonIntValue(String modelName, String version, String element, int defaultVal) {
        int value = defaultVal;
        if (this.modelConfig.containsKey(modelName)) {
            Map<String, JsonObject> versionModel = this.modelConfig.get(modelName);
            JsonObject jsonObject = versionModel.getOrDefault(version, null);

            if (jsonObject != null && jsonObject.get(element) != null) {
                try {
                    value = jsonObject.get(element).getAsInt();
                    if (value <= 0) {
                        value = defaultVal;
                    }
                } catch (ClassCastException | IllegalStateException e) {
                    LoggerFactory.getLogger(ConfigManager.class)
                            .error(
                                    "Invalid value for model: {}:{}, parameter: {}",
                                    modelName,
                                    version,
                                    element);
                    return defaultVal;
                }
            }
        }
        return value;
    }

    public Manifest.RuntimeType getJsonRuntimeTypeValue(
            String modelName, String version, String element, Manifest.RuntimeType defaultVal) {
        Manifest.RuntimeType value = defaultVal;
        if (this.modelConfig.containsKey(modelName)) {
            Map<String, JsonObject> versionModel = this.modelConfig.get(modelName);
            JsonObject jsonObject = versionModel.getOrDefault(version, null);

            if (jsonObject != null && jsonObject.get(element) != null) {
                try {
                    value = Manifest.RuntimeType.fromValue(jsonObject.get(element).getAsString());
                } catch (ClassCastException | IllegalStateException | IllegalArgumentException e) {
                    LoggerFactory.getLogger(ConfigManager.class)
                            .error(
                                    "Invalid value for model: {}:{}, parameter: {}",
                                    modelName,
                                    version,
                                    element);
                    return defaultVal;
                }
            }
        }
        return value;
    }

    public String getVersion() {
        return prop.getProperty(VERSION);
    }

    public boolean isTelemetryEnabled() {
        return telemetryEnabled;
    }

    public static final class Arguments {

        private String tsConfigFile;
        private String pythonExecutable;
        private String modelStore;
        private String[] models;
        private boolean snapshotDisabled;
        private String workflowStore;
        private String cppLogConfigFile;
        private boolean tokenAuthEnabled;
        private boolean modelApiEnabled;

        public Arguments() {}

        public Arguments(CommandLine cmd) {
            tsConfigFile = cmd.getOptionValue("ts-config-file");
            pythonExecutable = cmd.getOptionValue("python");
            modelStore = cmd.getOptionValue("model-store");
            models = cmd.getOptionValues("models");
            snapshotDisabled = cmd.hasOption("no-config-snapshot");
            workflowStore = cmd.getOptionValue("workflow-store");
            cppLogConfigFile = cmd.getOptionValue("cpp-log-config");
            tokenAuthEnabled = cmd.hasOption("disable-token-auth");
            modelApiEnabled = cmd.hasOption("enable-model-api");
        }

        public static Options getOptions() {
            Options options = new Options();
            options.addOption(
                    Option.builder("f")
                            .longOpt("ts-config-file")
                            .hasArg()
                            .argName("TS-CONFIG-FILE")
                            .desc("Path to the configuration properties file.")
                            .build());
            options.addOption(
                    Option.builder("e")
                            .longOpt("python")
                            .hasArg()
                            .argName("PYTHON")
                            .desc("Python runtime executable path.")
                            .build());
            options.addOption(
                    Option.builder("m")
                            .longOpt("models")
                            .hasArgs()
                            .argName("MODELS")
                            .desc("Models to be loaded at startup.")
                            .build());
            options.addOption(
                    Option.builder("s")
                            .longOpt("model-store")
                            .hasArg()
                            .argName("MODELS-STORE")
                            .desc("Model store location where models can be loaded.")
                            .build());
            options.addOption(
                    Option.builder("ncs")
                            .longOpt("no-config-snapshot")
                            .argName("NO-CONFIG-SNAPSHOT")
                            .desc("disable torchserve snapshot")
                            .build());
            options.addOption(
                    Option.builder("w")
                            .longOpt("workflow-store")
                            .hasArg()
                            .argName("WORKFLOW-STORE")
                            .desc("Workflow store location where workflow can be loaded.")
                            .build());
            options.addOption(
                    Option.builder("clog")
                            .longOpt("cpp-log-config")
                            .hasArg()
                            .argName("CPP-LOG-CONFIG")
                            .desc("log configuration file for cpp backend.")
                            .build());
            options.addOption(
                    Option.builder("dt")
                            .longOpt("disable-token-auth")
                            .argName("TOKEN")
                            .desc("disables token authorization")
                            .build());
            options.addOption(
                    Option.builder("mapi")
                            .longOpt("enable-model-api")
                            .argName("ENABLE-MODEL-API")
                            .desc("sets model apis to enabled")
                            .build());
            return options;
        }

        public String getTsConfigFile() {
            return tsConfigFile;
        }

        public String getPythonExecutable() {
            return pythonExecutable;
        }

        public void setTsConfigFile(String tsConfigFile) {
            this.tsConfigFile = tsConfigFile;
        }

        public String getModelStore() {
            return modelStore;
        }

        public String getWorkflowStore() {
            return workflowStore;
        }

        public String isTokenDisabled() {
            return tokenAuthEnabled ? "true" : "false";
        }

        public void setModelStore(String modelStore) {
            this.modelStore = modelStore;
        }

        public String[] getModels() {
            return models;
        }

        public void setModels(String[] models) {
            this.models = models.clone();
        }

        public String isModelEnabled() {
            return String.valueOf(modelApiEnabled);
        }

        public boolean isSnapshotDisabled() {
            return snapshotDisabled;
        }

        public void setSnapshotDisabled(boolean snapshotDisabled) {
            this.snapshotDisabled = snapshotDisabled;
        }

        public String getCppLogConfigFile() {
            return cppLogConfigFile;
        }

        public void setCppLogConfigFile(String cppLogConfigFile) {
            this.cppLogConfigFile = cppLogConfigFile;
        }
    }
}
