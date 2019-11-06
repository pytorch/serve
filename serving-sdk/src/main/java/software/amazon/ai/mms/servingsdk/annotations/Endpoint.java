/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 */
package software.amazon.ai.mms.servingsdk.annotations;

import software.amazon.ai.mms.servingsdk.annotations.helpers.EndpointTypes;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface Endpoint {
    /**
     * @return URL pattern to which this class applies
     */
    String urlPattern() default "";

    /**
     * @return Type of this endpoint. Default NONE
     */
    EndpointTypes endpointType() default EndpointTypes.NONE;

    /**
     * @return Description of this endpoint. Default ""
     */
    String description() default "";
}
