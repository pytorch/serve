/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package software.amazon.ai.mms.servingsdk;

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
