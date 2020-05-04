
package org.pytorch.serve.servingsdk;

import java.util.List;

/**
 * This provides information about the model which is currently registered with Model Server
 */
public interface Model {
    /**
     * Get the name of this model
     * @return The name of this model
     */
    String getModelName();

    /**
     * Get the URL of the Model location
     * @return models URL
     */
    String getModelUrl();

    /**
     * Get the model's entry-point
     * @return "handler" invoked to handle requests
     */
    String getModelHandler();

    /**
     * Returns the current list of workers for this model
     * @return list of Worker objects
     */
    List<Worker> getModelWorkers();
}
