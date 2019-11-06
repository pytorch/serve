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
package com.amazonaws.ml.mms.http.messages;

import com.amazonaws.ml.mms.util.ConfigManager;
import com.amazonaws.ml.mms.util.NettyUtils;
import com.google.gson.annotations.SerializedName;
import io.netty.handler.codec.http.QueryStringDecoder;

/** Register Model Request for Model server */
public class RegisterModelRequest {
    @SerializedName("model_name")
    private String modelName;

    @SerializedName("runtime")
    private String runtime;

    @SerializedName("handler")
    private String handler;

    @SerializedName("batch_size")
    private int batchSize;

    @SerializedName("max_batch_delay")
    private int maxBatchDelay;

    @SerializedName("initial_workers")
    private int initialWorkers;

    @SerializedName("synchronous")
    private boolean synchronous;

    @SerializedName("response_timeout")
    private int responseTimeout;

    @SerializedName("url")
    private String modelUrl;

    public RegisterModelRequest(QueryStringDecoder decoder) {
        modelName = NettyUtils.getParameter(decoder, "model_name", null);
        runtime = NettyUtils.getParameter(decoder, "runtime", null);
        handler = NettyUtils.getParameter(decoder, "handler", null);
        batchSize = NettyUtils.getIntParameter(decoder, "batch_size", 1);
        maxBatchDelay = NettyUtils.getIntParameter(decoder, "max_batch_delay", 100);
        initialWorkers =
                NettyUtils.getIntParameter(
                        decoder,
                        "initial_workers",
                        ConfigManager.getInstance().getConfiguredDefaultWorkersPerModel());
        synchronous = Boolean.parseBoolean(NettyUtils.getParameter(decoder, "synchronous", "true"));
        responseTimeout = NettyUtils.getIntParameter(decoder, "response_timeout", -1);
        modelUrl = NettyUtils.getParameter(decoder, "url", null);
    }

    public RegisterModelRequest() {
        batchSize = 1;
        maxBatchDelay = 100;
        synchronous = true;
        initialWorkers = ConfigManager.getInstance().getConfiguredDefaultWorkersPerModel();
        responseTimeout = -1;
    }

    public String getModelName() {
        return modelName;
    }

    public String getRuntime() {
        return runtime;
    }

    public String getHandler() {
        return handler;
    }

    public Integer getBatchSize() {
        return batchSize;
    }

    public Integer getMaxBatchDelay() {
        return maxBatchDelay;
    }

    public Integer getInitialWorkers() {
        return initialWorkers;
    }

    public Boolean getSynchronous() {
        return synchronous;
    }

    public Integer getResponseTimeout() {
        return responseTimeout;
    }

    public String getModelUrl() {
        return modelUrl;
    }
}
