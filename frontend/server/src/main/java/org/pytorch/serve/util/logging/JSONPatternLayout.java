package org.pytorch.serve.util.logging;

import org.apache.logging.log4j.core.Layout;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.config.Node;
import org.apache.logging.log4j.core.config.plugins.Plugin;
import org.apache.logging.log4j.core.config.plugins.PluginFactory;
import org.apache.logging.log4j.core.layout.AbstractStringLayout;
import org.apache.logging.log4j.message.Message;
import org.pytorch.serve.metrics.Metric;
import org.pytorch.serve.util.JsonUtils;

@Plugin(
        name = "JSONPatternLayout",
        category = Node.CATEGORY,
        elementType = Layout.ELEMENT_TYPE,
        printObject = true)
public class JSONPatternLayout extends AbstractStringLayout {
    public JSONPatternLayout() {
        super(null, null, null);
    }

    @PluginFactory
    public static JSONPatternLayout createLayout() {
        return new JSONPatternLayout();
    }

    @Override
    public String toSerializable(LogEvent event) {
        Message eventMessage = event.getMessage();
        if (eventMessage == null || eventMessage.getParameters() == null) {
            return null;
        }

        Object[] parameters = eventMessage.getParameters();
        for (Object obj : parameters) {
            if (obj instanceof Metric) {
                Metric metric = (Metric) obj;
                return JsonUtils.GSON_PRETTY.toJson(metric) + "\n";
            }
        }

        return eventMessage.toString() + '\n';
    }
}
