package org.pytorch.serve.util.logging;

import org.apache.log4j.PatternLayout;
import org.apache.log4j.spi.LoggingEvent;
import org.pytorch.serve.metrics.Dimension;
import org.pytorch.serve.metrics.Metric;

public class QLogLayout extends PatternLayout {

    /**
     * Model server also supports query log formatting.
     *
     * <p>To enable Query Log format, change the layout as follows
     *
     * <pre>
     *     log4j.appender.model_metrics.layout = QLogLayout
     * </pre>
     *
     * This enables logs which are shown as following
     *
     * <pre>
     *     HostName=hostName
     *     RequestId=004bd136-063c-4102-a070-d7aff5add939
     *     Marketplace=US
     *     StartTime=1542275707
     *     Program=MXNetModelServer
     *     Metrics=PredictionTime=45 Milliseconds ModelName|squeezenet  Level|Model
     *     EOE
     * </pre>
     *
     * <b>Note</b>: The following entities in this metrics can be customized.
     *
     * <ul>
     *   <li><b>Marketplace</b> : This can be customized by setting the "REALM" system environment
     *       variable.
     *   <li><b>Program</b> : This entity can be customized by setting "MXNETMODELSERVER_PROGRAM"
     *       environment variable.
     * </ul>
     *
     * Example: If the above environment variables are set to the following,
     *
     * <pre>
     *     $ env
     *     REALM=someRealm
     *     MXNETMODELSERVER_PROGRAM=someProgram
     * </pre>
     *
     * This produces the metrics as follows
     *
     * <pre>
     *    HostName=hostName
     *    RequestId=004bd136-063c-4102-a070-d7aff5add939
     *    Marketplace=someRealm
     *    StartTime=1542275707
     *    Program=someProgram
     *    Metrics=PredictionTime=45 Milliseconds ModelName|squeezenet  Level|Model
     *    EOE
     * </pre>
     *
     * @param event
     * @return
     */
    @Override
    public String format(LoggingEvent event) {
        Object eventMessage = event.getMessage();
        String programName =
                getStringOrDefault(System.getenv("MXNETMODELSERVER_PROGRAM"), "MXNetModelServer");
        String domain = getStringOrDefault(System.getenv("DOMAIN"), "Unknown");

        long currentTimeInSec = System.currentTimeMillis() / 1000;

        if (eventMessage == null) {
            return null;
        }
        if (eventMessage instanceof Metric) {
            String marketPlace = System.getenv("REALM");

            StringBuilder stringBuilder = new StringBuilder();
            Metric metric = (Metric) eventMessage;
            stringBuilder.append("HostName=").append(metric.getHostName());

            if (metric.getRequestId() != null && !metric.getRequestId().isEmpty()) {
                stringBuilder.append("\nRequestId=").append(metric.getRequestId());
            }

            // Marketplace format should be : <programName>:<domain>:<realm>
            if (marketPlace != null && !marketPlace.isEmpty()) {
                stringBuilder
                        .append("\nMarketplace=")
                        .append(programName)
                        .append(':')
                        .append(domain)
                        .append(':')
                        .append(marketPlace);
            }

            stringBuilder
                    .append("\nStartTime=")
                    .append(
                            getStringOrDefault(
                                    metric.getTimestamp(), Long.toString(currentTimeInSec)));

            stringBuilder
                    .append("\nProgram=")
                    .append(programName)
                    .append("\nMetrics=")
                    .append(metric.getMetricName())
                    .append('=')
                    .append(metric.getValue())
                    .append(' ')
                    .append(metric.getUnit());
            for (Dimension dimension : metric.getDimensions()) {
                stringBuilder
                        .append(' ')
                        .append(dimension.getName())
                        .append('|')
                        .append(dimension.getValue())
                        .append(' ');
            }
            stringBuilder.append("\nEOE\n");

            return stringBuilder.toString();
        }
        return eventMessage.toString();
    }

    private static String getStringOrDefault(String val, String defVal) {

        if (val == null || val.isEmpty()) {
            return defVal;
        }
        return val;
    }
}
