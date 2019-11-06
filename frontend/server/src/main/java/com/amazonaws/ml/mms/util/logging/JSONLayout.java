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

package com.amazonaws.ml.mms.util.logging;

import com.amazonaws.ml.mms.metrics.Metric;
import com.amazonaws.ml.mms.util.JsonUtils;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.spi.LoggingEvent;

public class JSONLayout extends PatternLayout {

    @Override
    public String format(LoggingEvent event) {
        Object eventMessage = event.getMessage();
        if (eventMessage == null) {
            return null;
        }
        if (eventMessage instanceof Metric) {
            Metric metric = (Metric) event.getMessage();
            return JsonUtils.GSON_PRETTY.toJson(metric) + '\n';
        }
        return eventMessage.toString() + '\n';
    }
}
