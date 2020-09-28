package org.pytorch.serve.metrics;

import com.google.gson.annotations.SerializedName;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Getter
@Setter
public class Metric {

    private static final Pattern PATTERN =
            Pattern.compile(
                    "\\s*([\\w\\s]+)\\.([\\w\\s]+):([0-9\\-,.e]+)\\|#([^|]*)\\|#hostname:([^,]+),([^,]+)(,(.*))?");

    @SerializedName("MetricName")
    private String metricName;

    @SerializedName("Value")
    private String value;

    @SerializedName("Unit")
    private String unit;

    @SerializedName("Dimensions")
    private List<Dimension> dimensions;

    @SerializedName("Timestamp")
    private String timestamp;

    @SerializedName("RequestId")
    private String requestId;

    @SerializedName("HostName")
    private String hostName;

    public Metric() {}

    public Metric(
            String metricName,
            String value,
            String unit,
            String hostName,
            Dimension... dimensions) {
        this.metricName = metricName;
        this.value = value;
        this.unit = unit;
        this.hostName = hostName;
        this.dimensions = Arrays.asList(dimensions);
    }

    public static Metric parse(String line) {
        // DiskAvailable.Gigabytes:311|#Level:Host,hostname:localhost
        Matcher matcher = PATTERN.matcher(line);
        if (!matcher.matches()) {
            return null;
        }

        Metric metric = new Metric();
        metric.setMetricName(matcher.group(1));
        metric.setUnit(matcher.group(2));
        metric.setValue(matcher.group(3));
        String dimensions = matcher.group(4);
        metric.setHostName(matcher.group(5));
        metric.setTimestamp(matcher.group(6));
        metric.setRequestId(matcher.group(8));

        if (dimensions != null) {
            String[] dimension = dimensions.split(",");
            List<Dimension> list = new ArrayList<>(dimension.length);
            for (String dime : dimension) {
                String[] pair = dime.split(":");
                if (pair.length == 2) {
                    list.add(new Dimension(pair[0], pair[1]));
                }
            }
            metric.setDimensions(list);
        }

        return metric;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append(metricName).append('.').append(unit).append(':').append(getValue()).append("|#");
        boolean first = true;
        for (Dimension dimension : getDimensions()) {
            if (first) {
                first = false;
            } else {
                sb.append(',');
            }
            sb.append(dimension.getName()).append(':').append(dimension.getValue());
        }
        sb.append("|#hostname:").append(hostName);
        if (requestId != null) {
            sb.append(",requestID:").append(requestId);
        }
        sb.append(",timestamp:").append(timestamp);
        return sb.toString();
    }
}
