/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package com.amazonaws.ml.mms.util.messages;

import com.amazonaws.ml.mms.wlm.Model;

public class ModelLoadModelRequest extends BaseModelRequest {

    /**
     * ModelLoadModelRequest is a interface between frontend and backend to notify the backend to
     * load a particular model.
     */
    private String modelPath;

    private String handler;
    private int batchSize;
    private int gpuId;

    public ModelLoadModelRequest(Model model, int gpuId) {
        super(WorkerCommands.LOAD, model.getModelName());
        this.gpuId = gpuId;
        modelPath = model.getModelDir().getAbsolutePath();
        handler = model.getModelArchive().getManifest().getModel().getHandler();
        batchSize = model.getBatchSize();
    }

    public String getModelPath() {
        return modelPath;
    }

    public String getHandler() {
        return handler;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getGpuId() {
        return gpuId;
    }
}
