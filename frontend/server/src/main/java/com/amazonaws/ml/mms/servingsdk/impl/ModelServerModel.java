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

import com.amazonaws.ml.mms.wlm.ModelManager;
import java.util.ArrayList;
import java.util.List;
import software.amazon.ai.mms.servingsdk.Model;
import software.amazon.ai.mms.servingsdk.Worker;

public class ModelServerModel implements Model {
    private final com.amazonaws.ml.mms.wlm.Model model;

    public ModelServerModel(com.amazonaws.ml.mms.wlm.Model m) {
        model = m;
    }

    @Override
    public String getModelName() {
        return model.getModelName();
    }

    @Override
    public String getModelUrl() {
        return model.getModelUrl();
    }

    @Override
    public String getModelHandler() {
        return model.getModelArchive().getHandler();
    }

    @Override
    public List<Worker> getModelWorkers() {
        ArrayList<Worker> list = new ArrayList<>();
        ModelManager.getInstance()
                .getWorkers(model.getModelName())
                .forEach(r -> list.add(new ModelWorker(r)));
        return list;
    }
}
