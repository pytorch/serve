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

import java.util.ArrayList;
import java.util.List;

public class ModelInferenceRequest extends BaseModelRequest {

    private List<RequestInput> batch;

    public ModelInferenceRequest(String modelName) {
        super(WorkerCommands.PREDICT, modelName);
        batch = new ArrayList<>();
    }

    public List<RequestInput> getRequestBatch() {
        return batch;
    }

    public void setRequestBatch(List<RequestInput> requestBatch) {
        this.batch = requestBatch;
    }

    public void addRequest(RequestInput req) {
        batch.add(req);
    }
}
