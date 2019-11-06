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

import java.util.LinkedHashMap;
import java.util.Map;

public class Response {

    private transient String code;
    private String description;
    private Map<String, MediaType> content;

    public Response() {}

    public Response(String code, String description) {
        this.code = code;
        this.description = description;
    }

    public Response(String code, String description, MediaType mediaType) {
        this.code = code;
        this.description = description;
        content = new LinkedHashMap<>();
        content.put(mediaType.getContentType(), mediaType);
    }

    public String getCode() {
        return code;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Map<String, MediaType> getContent() {
        return content;
    }

    public void setContent(Map<String, MediaType> content) {
        this.content = content;
    }

    public void addContent(MediaType mediaType) {
        if (content == null) {
            content = new LinkedHashMap<>();
        }
        content.put(mediaType.getContentType(), mediaType);
    }
}
