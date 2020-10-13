
package org.pytorch.serve.servingsdk.metrics;


/**
 * This class defines the abstract class for public abstract class MetricListenerRegistry {
 */
public abstract class MetricEventListenerRegistry {

    /**
     * This method is called when a HTTP DELETE method is invoked for the defined custom model server endpoint
     * @param mep - MetricEventPublisher to register the listener
     */
    public void register(MetricEventPublisher mep){
        throw new MetricPluginException("No implementation found .. Default implementation invoked");

    }
}

