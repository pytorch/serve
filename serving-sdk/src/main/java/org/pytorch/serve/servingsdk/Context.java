
package org.pytorch.serve.servingsdk;

import java.util.Map;
import java.util.Properties;

/**
 * This interface provides access to the current running Model Server.
 */
public interface Context {
    /**
     * Get the configuration of the current running Model Server
     * @return Properties
     */
    Properties getConfig();

    /**
     * Get a list of Models registered with the Model Server
     * @return List of models
     */
    Map<String, Model> getModels();
}
