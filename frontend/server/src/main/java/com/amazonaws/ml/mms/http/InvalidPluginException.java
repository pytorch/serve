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
package com.amazonaws.ml.mms.http;

/** InvaliPluginException is thrown when there is an error while handling a Model Server plugin */
public class InvalidPluginException extends RuntimeException {

    static final long serialVersionUID = 1L;
    /**
     * Constructs an {@code InvalidPluginException} with {@code null} as its error detail message.
     */
    public InvalidPluginException() {
        super("Registered plugin is invalid. Please re-check the configuration and the plugins.");
    }

    /**
     * Constructs an {@code InvalidPluginException} with {@code msg} as its error detail message
     *
     * @param msg : This is the error detail message
     */
    public InvalidPluginException(String msg) {
        super(msg);
    }
}
