
package org.pytorch.serve.servingsdk.annotations;

import org.pytorch.serve.servingsdk.annotations.helpers.EndpointTypes;

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
