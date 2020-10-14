package org.pytorch.serve.servingsdk.impl;

import java.lang.annotation.Annotation;
import java.util.HashMap;
import java.util.Map;
import java.util.ServiceLoader;
import org.pytorch.serve.http.InvalidPluginException;
import org.pytorch.serve.metrics.plugin.MetricEventPublisherImpl;
import org.pytorch.serve.servingsdk.ModelServerEndpoint;
import org.pytorch.serve.servingsdk.annotations.Endpoint;
import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;
import org.pytorch.serve.servingsdk.metrics.MetricEventListenerRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class PluginsManager {

    private static final PluginsManager INSTANCE = new PluginsManager();
    private Logger logger = LoggerFactory.getLogger(PluginsManager.class);

    private Map<String, ModelServerEndpoint> inferenceEndpoints;
    private Map<String, ModelServerEndpoint> managementEndpoints;
    private Map<String, ModelServerEndpoint> metricEndpoints;

    private PluginsManager() {}

    public static PluginsManager getInstance() {
        return INSTANCE;
    }

    public void initialize() {
        inferenceEndpoints = initInferenceEndpoints();
        managementEndpoints = initManagementEndpoints();
        metricEndpoints = initMetricEndpoints();
        initializeMetricListeners();
    }

    private boolean validateEndpointPlugin(Annotation a, EndpointTypes type) {
        return a instanceof Endpoint
                && !((Endpoint) a).urlPattern().isEmpty()
                && ((Endpoint) a).endpointType().equals(type);
    }

    private HashMap<String, ModelServerEndpoint> getEndpoints(EndpointTypes type)
            throws InvalidPluginException {
        ServiceLoader<ModelServerEndpoint> loader = ServiceLoader.load(ModelServerEndpoint.class);
        HashMap<String, ModelServerEndpoint> ep = new HashMap<>();
        for (ModelServerEndpoint mep : loader) {
            Class<? extends ModelServerEndpoint> modelServerEndpointClassObj = mep.getClass();
            Annotation[] annotations = modelServerEndpointClassObj.getAnnotations();
            for (Annotation a : annotations) {
                if (validateEndpointPlugin(a, type)) {
                    if (ep.get(((Endpoint) a).urlPattern()) != null) {
                        throw new InvalidPluginException(
                                "Multiple plugins found for endpoint "
                                        + "\""
                                        + ((Endpoint) a).urlPattern()
                                        + "\"");
                    }
                    logger.info("Loading plugin for endpoint {}", ((Endpoint) a).urlPattern());
                    ep.put(((Endpoint) a).urlPattern(), mep);
                }
            }
        }
        return ep;
    }

    private void initializeMetricListeners() throws InvalidPluginException {
        ServiceLoader<MetricEventListenerRegistry> loader =
                ServiceLoader.load(MetricEventListenerRegistry.class);
        for (MetricEventListenerRegistry registry : loader) {
            Class<? extends MetricEventListenerRegistry> registryClass = registry.getClass();
            logger.info(
                    "Registering metric listener for plugin class {}.", registryClass.getName());
            registry.register(MetricEventPublisherImpl.getInstance());
        }
    }

    private HashMap<String, ModelServerEndpoint> initInferenceEndpoints() {
        return getEndpoints(EndpointTypes.INFERENCE);
    }

    private HashMap<String, ModelServerEndpoint> initManagementEndpoints() {
        return getEndpoints(EndpointTypes.MANAGEMENT);
    }

    private HashMap<String, ModelServerEndpoint> initMetricEndpoints() {
        return getEndpoints(EndpointTypes.METRIC);
    }

    public Map<String, ModelServerEndpoint> getInferenceEndpoints() {
        return inferenceEndpoints;
    }

    public Map<String, ModelServerEndpoint> getManagementEndpoints() {
        return managementEndpoints;
    }

    public Map<String, ModelServerEndpoint> getMetricEndPoints() {
        return metricEndpoints;
    }
}
