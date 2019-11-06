/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved. Licensed under the Apache
 * License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of
 * the License is located at http://aws.amazon.com/apache2.0/ or in the "license" file accompanying this file. This file
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissionsand limitations under the License.
 */

package software.amazon.ai.mms.servingsdk;

/**
 * Describe the model worker
 */
public interface Worker {
    /**
     * Get the current running status of this model's worker
     * @return True - if the worker is currently running. False - the worker is currently not running.
     */
    boolean isRunning();

    /**
     * Get the current memory foot print of this worker
     * @return Current memory usage of this worker
     */
    long getWorkerMemory();
}
