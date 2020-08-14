package org.pytorch.serve.util;

import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.group.ChannelGroup;
import io.netty.channel.group.ChannelGroupFuture;
import io.netty.channel.group.DefaultChannelGroup;
import io.netty.util.concurrent.GlobalEventExecutor;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ServerGroups {

    private static final Logger logger = LoggerFactory.getLogger(ServerGroups.class);

    private ChannelGroup allChannels;

    private EventLoopGroup serverGroup;
    private EventLoopGroup childGroup;
    private EventLoopGroup metricsGroup;
    private EventLoopGroup backendGroup;

    private ConfigManager configManager;

    public ServerGroups(ConfigManager configManager) {
        this.configManager = configManager;
        init();
    }

    public final void init() {
        allChannels = new DefaultChannelGroup(GlobalEventExecutor.INSTANCE);

        serverGroup = Connector.newEventLoopGroup(2);
        childGroup = Connector.newEventLoopGroup(configManager.getNettyThreads());
        if (configManager.isMetricApiEnable()) {
            metricsGroup = Connector.newEventLoopGroup(1);
        }
        backendGroup = Connector.newEventLoopGroup(configManager.getNettyClientThreads());
    }

    public void shutdown(boolean graceful) {
        closeAllChannels(graceful);

        List<EventLoopGroup> allEventLoopGroups = new ArrayList<>();

        allEventLoopGroups.add(serverGroup);
        allEventLoopGroups.add(childGroup);
        if (configManager.isMetricApiEnable()) {
            allEventLoopGroups.add(metricsGroup);
        }

        for (EventLoopGroup group : allEventLoopGroups) {
            if (graceful) {
                group.shutdownGracefully();
            } else {
                group.shutdownGracefully(0, 0, TimeUnit.SECONDS);
            }
        }

        if (graceful) {
            for (EventLoopGroup group : allEventLoopGroups) {
                try {
                    group.awaitTermination(60, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    public EventLoopGroup getServerGroup() {
        return serverGroup;
    }

    public EventLoopGroup getChildGroup() {
        return childGroup;
    }

    public EventLoopGroup getMetricsGroup() {
        return metricsGroup;
    }

    public EventLoopGroup getBackendGroup() {
        return backendGroup;
    }

    public void registerChannel(Channel channel) {
        allChannels.add(channel);
    }

    private void closeAllChannels(boolean graceful) {
        ChannelGroupFuture future = allChannels.close();

        // if this is a graceful shutdown, log any channel closing failures. if this isn't a
        // graceful shutdown, ignore them.
        if (graceful) {
            try {
                future.await(10, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            if (!future.isSuccess()) {
                for (ChannelFuture cf : future) {
                    if (!cf.isSuccess()) {
                        logger.info("Unable to close channel: " + cf.channel(), cf.cause());
                    }
                }
            }
        }
    }
}
