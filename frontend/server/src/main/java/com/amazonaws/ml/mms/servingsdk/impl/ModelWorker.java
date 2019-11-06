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

package com.amazonaws.ml.mms.servingsdk.impl;

import com.amazonaws.ml.mms.wlm.WorkerState;
import com.amazonaws.ml.mms.wlm.WorkerThread;
import software.amazon.ai.mms.servingsdk.Worker;

public class ModelWorker implements Worker {
    boolean running;
    long memory;

    public ModelWorker(WorkerThread t) {
        running = t.getState() == WorkerState.WORKER_MODEL_LOADED;
        memory = t.getMemory();
    }

    @Override
    public boolean isRunning() {
        return running;
    }

    @Override
    public long getWorkerMemory() {
        return memory;
    }
}
