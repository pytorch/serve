
package org.pytorch.serve.servingsdk.metrics;


/**
 * This class defines the abstract class to facilitate the registration of plugin's MetricEventListener
 */
public abstract class MetricEventListenerRegistry {

    /**
     * This method is called at the time of TorchServe initialization to register the MetricEventListener with
     * MetricEventPublisher.
     * @param mep - MetricEventPublisher to register the listener
     */
    public void register(MetricEventPublisher mep){
        throw new MetricPluginException("No implementation found .. Default implementation invoked");

    }
}

