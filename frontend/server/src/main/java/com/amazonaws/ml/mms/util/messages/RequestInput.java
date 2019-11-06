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

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RequestInput {

    private String requestId;
    private Map<String, String> headers;
    private List<InputParameter> parameters;

    public RequestInput(String requestId) {
        this.requestId = requestId;
        headers = new HashMap<>();
        parameters = new ArrayList<>();
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public Map<String, String> getHeaders() {
        return headers;
    }

    public void setHeaders(Map<String, String> headers) {
        this.headers = headers;
    }

    public void updateHeaders(String key, String val) {
        headers.put(key, val);
    }

    public List<InputParameter> getParameters() {
        return parameters;
    }

    public void setParameters(List<InputParameter> parameters) {
        this.parameters = parameters;
    }

    public void addParameter(InputParameter modelInput) {
        parameters.add(modelInput);
    }

    public String getStringParameter(String key) {
        for (InputParameter param : parameters) {
            if (key.equals(param.getName())) {
                return new String(param.getValue(), StandardCharsets.UTF_8);
            }
        }
        return null;
    }
}
