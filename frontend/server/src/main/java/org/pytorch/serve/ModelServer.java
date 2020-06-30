package org.pytorch.serve;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.FixedRecvByteBufAllocator;
import io.netty.channel.ServerChannel;
import io.netty.handler.ssl.SslContext;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.File;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.InvalidPropertiesFormatException;
import java.util.List;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.pytorch.serve.archive.ModelArchive;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.metrics.MetricManager;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;
import org.pytorch.serve.servingsdk.impl.PluginsManager;
import org.pytorch.serve.snapshot.InvalidSnapshotException;
import org.pytorch.serve.snapshot.SnapshotManager;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.util.Connector;
import org.pytorch.serve.util.ConnectorType;
import org.pytorch.serve.util.ServerGroups;
import org.pytorch.serve.wlm.ModelManager;
import org.pytorch.serve.wlm.WorkLoadManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelServer {

    private Logger logger = LoggerFactory.getLogger(ModelServer.class);

    private ServerGroups serverGroups;
    private List<ChannelFuture> futures = new ArrayList<>(2);
    private AtomicBoolean stopped = new AtomicBoolean(false);
    private ConfigManager configManager;
    public static final int MAX_RCVBUF_SIZE = 4096;

    /** Creates a new {@code ModelServer} instance. */
    public ModelServer(ConfigManager configManager) {
        this.configManager = configManager;
        serverGroups = new ServerGroups(configManager);
    }

    public static void main(String[] args) {
        Options options = ConfigManager.Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            ConfigManager.Arguments arguments = new ConfigManager.Arguments(cmd);
            ConfigManager.init(arguments);
            ConfigManager configManager = ConfigManager.getInstance();
            PluginsManager.getInstance().initialize();
            InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);
            ModelServer modelServer = new ModelServer(configManager);

            Runtime.getRuntime()
                    .addShutdownHook(
                            new Thread() {
                                @Override
                                public void run() {
                                    modelServer.stop();
                                }
                            });

            modelServer.startAndWait();
        } catch (IllegalArgumentException e) {
            System.out.println("Invalid configuration: " + e.getMessage()); // NOPMD
        } catch (ParseException e) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setLeftPadding(1);
            formatter.setWidth(120);
            formatter.printHelp(e.getMessage(), options);
        } catch (Throwable t) {
            t.printStackTrace(); // NOPMD
        } finally {
            System.exit(1); // NOPMD
        }
    }

    public void startAndWait()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        try {
            List<ChannelFuture> channelFutures = start();
            // Create and schedule metrics manager
            MetricManager.scheduleMetrics(configManager);
            System.out.println("Model server started."); // NOPMD

            channelFutures.get(0).sync();
        } catch (InvalidPropertiesFormatException e) {
            logger.error("Invalid configuration", e);
        } finally {
            serverGroups.shutdown(true);
            logger.info("Torchserve stopped.");
        }
    }

    private String getDefaultModelName(String name) {
        if (name.contains(".model") || name.contains(".mar")) {
            return name.substring(name.lastIndexOf('/') + 1, name.lastIndexOf('.'))
                    .replaceAll("(\\W|^_)", "_");
        } else {
            return name.substring(name.lastIndexOf('/') + 1).replaceAll("(\\W|^_)", "_");
        }
    }

    private void initModelStore() throws InvalidSnapshotException, IOException {
        WorkLoadManager wlm = new WorkLoadManager(configManager, serverGroups.getBackendGroup());
        ModelManager.init(configManager, wlm);
        SnapshotManager.init(configManager);
        Set<String> startupModels = ModelManager.getInstance().getStartupModels();
        String defaultModelName;
        String modelSnapshot = configManager.getModelSnapshot();
        if (modelSnapshot != null) {
            SnapshotManager.getInstance().restore(modelSnapshot);
            return;
        }

        String loadModels = configManager.getLoadModels();
        if (loadModels == null || loadModels.isEmpty()) {
            return;
        }

        ModelManager modelManager = ModelManager.getInstance();
        int workers = configManager.getDefaultWorkers();
        if ("ALL".equalsIgnoreCase(loadModels)) {
            String modelStore = configManager.getModelStore();
            if (modelStore == null) {
                logger.warn("Model store is not configured.");
                return;
            }

            File modelStoreDir = new File(modelStore);
            if (!modelStoreDir.exists()) {
                logger.warn("Model store path is not found: {}", modelStore);
                return;
            }

            // Check folders to see if they can be models as well
            File[] files = modelStoreDir.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isHidden()) {
                        continue;
                    }
                    String fileName = file.getName();
                    if (file.isFile()
                            && !fileName.endsWith(".mar")
                            && !fileName.endsWith(".model")) {
                        continue;
                    }
                    try {
                        logger.debug("Loading models from model store: {}", file.getName());
                        defaultModelName = getDefaultModelName(fileName);

                        ModelArchive archive =
                                modelManager.registerModel(file.getName(), defaultModelName);
                        modelManager.updateModel(
                                archive.getModelName(),
                                archive.getModelVersion(),
                                workers,
                                workers,
                                true);
                        startupModels.add(archive.getModelName());
                    } catch (ModelException | IOException e) {
                        logger.warn("Failed to load model: " + file.getAbsolutePath(), e);
                    }
                }
            }
            return;
        }

        String[] models = loadModels.split(",");
        for (String model : models) {
            String[] pair = model.split("=", 2);
            String modelName = null;
            String url;
            if (pair.length == 1) {
                url = pair[0];
            } else {
                modelName = pair[0];
                url = pair[1];
            }
            if (url.isEmpty()) {
                continue;
            }

            try {
                logger.info("Loading initial models: {}", url);
                defaultModelName = getDefaultModelName(url);

                ModelArchive archive =
                        modelManager.registerModel(
                                url,
                                modelName,
                                null,
                                null,
                                1,
                                100,
                                configManager.getDefaultResponseTimeout(),
                                defaultModelName);
                modelManager.updateModel(
                        archive.getModelName(), archive.getModelVersion(), workers, workers, true);
                startupModels.add(archive.getModelName());
            } catch (ModelException | IOException e) {
                logger.warn("Failed to load model: " + url, e);
            }
        }
    }

    public ChannelFuture initializeServer(
            Connector connector,
            EventLoopGroup serverGroup,
            EventLoopGroup workerGroup,
            ConnectorType type)
            throws InterruptedException, IOException, GeneralSecurityException {
        final String purpose = connector.getPurpose();
        Class<? extends ServerChannel> channelClass = connector.getServerChannel();
        logger.info("Initialize {} server with: {}.", purpose, channelClass.getSimpleName());
        ServerBootstrap b = new ServerBootstrap();
        b.option(ChannelOption.SO_BACKLOG, 1024)
                .channel(channelClass)
                .childOption(ChannelOption.SO_LINGER, 0)
                .childOption(ChannelOption.SO_REUSEADDR, true)
                .childOption(ChannelOption.SO_KEEPALIVE, true)
                .childOption(
                        ChannelOption.RCVBUF_ALLOCATOR,
                        new FixedRecvByteBufAllocator(MAX_RCVBUF_SIZE));
        b.group(serverGroup, workerGroup);

        SslContext sslCtx = null;
        if (connector.isSsl()) {
            sslCtx = configManager.getSslContext();
        }
        b.childHandler(new ServerInitializer(sslCtx, type));

        ChannelFuture future;
        try {
            future = b.bind(connector.getSocketAddress()).sync();
        } catch (Exception e) {
            // https://github.com/netty/netty/issues/2597
            if (e instanceof IOException) {
                throw new IOException("Failed to bind to address: " + connector, e);
            }
            throw e;
        }
        future.addListener(
                (ChannelFutureListener)
                        f -> {
                            if (!f.isSuccess()) {
                                try {
                                    f.get();
                                } catch (InterruptedException | ExecutionException e) {
                                    logger.error("", e);
                                }
                                System.exit(-1); // NO PMD
                            }
                            serverGroups.registerChannel(f.channel());
                        });

        future.sync();

        ChannelFuture f = future.channel().closeFuture();
        f.addListener(
                (ChannelFutureListener)
                        listener -> logger.info("{} model server stopped.", purpose));

        logger.info("{} API bind to: {}", purpose, connector);
        return f;
    }

    /**
     * Main Method that prepares the future for the channel and sets up the ServerBootstrap.
     *
     * @return A ChannelFuture object
     * @throws InterruptedException if interrupted
     * @throws InvalidSnapshotException
     */
    public List<ChannelFuture> start()
            throws InterruptedException, IOException, GeneralSecurityException,
                    InvalidSnapshotException {
        stopped.set(false);

        configManager.validateConfigurations();

        logger.info(configManager.dumpConfigurations());

        initModelStore();

        Connector inferenceConnector = configManager.getListener(false);
        Connector managementConnector = configManager.getListener(true);

        inferenceConnector.clean();
        managementConnector.clean();

        EventLoopGroup serverGroup = serverGroups.getServerGroup();
        EventLoopGroup workerGroup = serverGroups.getChildGroup();

        futures.clear();

        if (!inferenceConnector.equals(managementConnector)) {
            futures.add(
                    initializeServer(
                            inferenceConnector,
                            serverGroup,
                            workerGroup,
                            ConnectorType.INFERENCE_CONNECTOR));
            futures.add(
                    initializeServer(
                            managementConnector,
                            serverGroup,
                            workerGroup,
                            ConnectorType.MANAGEMENT_CONNECTOR));
        } else {
            futures.add(
                    initializeServer(
                            inferenceConnector, serverGroup, workerGroup, ConnectorType.BOTH));
        }

        SnapshotManager.getInstance().saveStartupSnapshot();
        return futures;
    }

    private boolean validEndpoint(Annotation a, EndpointTypes type) {
        return a instanceof Endpoint
                && !((Endpoint) a).urlPattern().isEmpty()
                && ((Endpoint) a).endpointType().equals(type);
    }

    private HashMap<String, ModelServerEndpoint> registerEndpoints(EndpointTypes type) {
        ServiceLoader<ModelServerEndpoint> loader = ServiceLoader.load(ModelServerEndpoint.class);
        HashMap<String, ModelServerEndpoint> ep = new HashMap<>();
        for (ModelServerEndpoint mep : loader) {
            Class<? extends ModelServerEndpoint> modelServerEndpointClassObj = mep.getClass();
            Annotation[] annotations = modelServerEndpointClassObj.getAnnotations();
            for (Annotation a : annotations) {
                if (validEndpoint(a, type)) {
                    ep.put(((Endpoint) a).urlPattern(), mep);
                }
            }
        }
        return ep;
    }

    public boolean isRunning() {
        return !stopped.get();
    }

    public void stop() {
        if (stopped.get()) {
            return;
        }

        stopped.set(true);
        for (ChannelFuture future : futures) {
            future.channel().close();
        }

        SnapshotManager.getInstance().saveShutdownSnapshot();
        serverGroups.shutdown(true);
        serverGroups.init();
    }
}
