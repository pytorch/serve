package org.pytorch.serve.util;

import io.netty.channel.Channel;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.ServerChannel;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDomainSocketChannel;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.epoll.EpollServerDomainSocketChannel;
import io.netty.channel.epoll.EpollServerSocketChannel;
import io.netty.channel.epoll.EpollSocketChannel;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueDomainSocketChannel;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.kqueue.KQueueServerDomainSocketChannel;
import io.netty.channel.kqueue.KQueueServerSocketChannel;
import io.netty.channel.kqueue.KQueueSocketChannel;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.channel.unix.DomainSocketAddress;
import java.io.File;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.util.Objects;
import java.util.regex.Matcher;
import org.apache.commons.io.FileUtils;

public class Connector {

    private static boolean useNativeIo = ConfigManager.getInstance().useNativeIo();

    private boolean uds;
    private String socketPath;
    private String bindIp;
    private int port;
    private boolean ssl;
    private ConnectorType connectorType;

    public Connector(int port) {
        this(port, useNativeIo && (Epoll.isAvailable() || KQueue.isAvailable()));
    }

    private Connector(int port, boolean uds) {
        this.port = port;
        this.uds = uds;
        if (uds) {
            bindIp = "";
            socketPath = System.getProperty("java.io.tmpdir") + "/.ts.sock." + port;
        } else {
            bindIp = "127.0.0.1";
            socketPath = String.valueOf(port);
        }
    }

    private Connector(
            int port,
            boolean uds,
            String bindIp,
            String socketPath,
            boolean ssl,
            ConnectorType connectorType) {
        this.port = port;
        this.uds = uds;
        this.bindIp = bindIp;
        this.socketPath = socketPath;
        this.ssl = ssl;
        this.connectorType = connectorType;
    }

    public static Connector parse(String binding, ConnectorType connectorType) {
        Matcher matcher = ConfigManager.ADDRESS_PATTERN.matcher(binding);
        if (!matcher.matches()) {
            throw new IllegalArgumentException("Invalid binding address: " + binding);
        }

        boolean uds = matcher.group(7) != null;
        if (uds) {
            if (!useNativeIo) {
                throw new IllegalArgumentException(
                        "unix domain socket requires use_native_io set to true.");
            }
            String path = matcher.group(7);
            return new Connector(-1, true, "", path, false, ConnectorType.MANAGEMENT_CONNECTOR);
        }

        String protocol = matcher.group(2);
        String host = matcher.group(3);
        String listeningPort = matcher.group(5);

        boolean ssl = "https".equalsIgnoreCase(protocol);
        int port;
        if (listeningPort == null) {
            switch (connectorType) {
                case MANAGEMENT_CONNECTOR:
                    port = ssl ? 8444 : 8081;
                    break;
                case METRICS_CONNECTOR:
                    port = ssl ? 8445 : 8082;
                    break;
                default:
                    port = ssl ? 443 : 80;
            }
        } else {
            port = Integer.parseInt(listeningPort);
        }
        if (port >= Short.MAX_VALUE * 2 + 1) {
            throw new IllegalArgumentException("Invalid port number: " + binding);
        }
        return new Connector(port, false, host, String.valueOf(port), ssl, connectorType);
    }

    public String getSocketType() {
        return uds ? "unix" : "tcp";
    }

    public String getSocketPath() {
        return socketPath;
    }

    public boolean isUds() {
        return uds;
    }

    public boolean isSsl() {
        return ssl;
    }

    public boolean isManagement() {
        return connectorType.equals(ConnectorType.MANAGEMENT_CONNECTOR);
    }

    public SocketAddress getSocketAddress() {
        return uds ? new DomainSocketAddress(socketPath) : new InetSocketAddress(bindIp, port);
    }

    public String getPurpose() {
        switch (connectorType) {
            case MANAGEMENT_CONNECTOR:
                return "Management";
            case METRICS_CONNECTOR:
                return "Metrics";
            default:
                return "Inference";
        }
    }

    public static EventLoopGroup newEventLoopGroup(int threads) {
        if (useNativeIo && Epoll.isAvailable()) {
            return new EpollEventLoopGroup(threads);
        } else if (useNativeIo && KQueue.isAvailable()) {
            return new KQueueEventLoopGroup(threads);
        }

        NioEventLoopGroup eventLoopGroup = new NioEventLoopGroup(threads);
        eventLoopGroup.setIoRatio(ConfigManager.getInstance().getIoRatio());
        return eventLoopGroup;
    }

    public Class<? extends ServerChannel> getServerChannel() {
        if (useNativeIo && Epoll.isAvailable()) {
            return uds ? EpollServerDomainSocketChannel.class : EpollServerSocketChannel.class;
        } else if (useNativeIo && KQueue.isAvailable()) {
            return uds ? KQueueServerDomainSocketChannel.class : KQueueServerSocketChannel.class;
        }

        return NioServerSocketChannel.class;
    }

    public Class<? extends Channel> getClientChannel() {
        if (useNativeIo && Epoll.isAvailable()) {
            return uds ? EpollDomainSocketChannel.class : EpollSocketChannel.class;
        } else if (useNativeIo && KQueue.isAvailable()) {
            return uds ? KQueueDomainSocketChannel.class : KQueueSocketChannel.class;
        }

        return NioSocketChannel.class;
    }

    public void clean() {
        if (uds) {
            FileUtils.deleteQuietly(new File(socketPath));
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Connector connector = (Connector) o;
        return uds == connector.uds
                && port == connector.port
                && socketPath.equals(connector.socketPath)
                && bindIp.equals(connector.bindIp);
    }

    @Override
    public int hashCode() {
        return Objects.hash(uds, socketPath, bindIp, port);
    }

    @Override
    public String toString() {
        if (uds) {
            return "unix:" + socketPath;
        } else if (ssl) {
            return "https://" + bindIp + ':' + port;
        }
        return "http://" + bindIp + ':' + port;
    }
}
