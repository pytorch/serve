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
package com.amazonaws.ml.mms.openapi;

public class QueryParameter extends Parameter {

    public QueryParameter() {
        this(null, "string", null, false, null);
    }

    public QueryParameter(String name, String description) {
        this(name, "string", null, false, description);
    }

    public QueryParameter(String name, String type, String description) {
        this(name, type, null, false, description);
    }

    public QueryParameter(String name, String type, String defaultValue, String description) {
        this(name, type, defaultValue, false, description);
    }

    public QueryParameter(
            String name, String type, String defaultValue, boolean required, String description) {
        this.name = name;
        this.description = description;
        in = "query";
        this.required = required;
        schema = new Schema(type, null, defaultValue);
    }
}
