package org.pytorch.serve.device.interfaces;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import org.pytorch.serve.device.Accelerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public interface IJsonSmiParser {
    static final Logger jsonSmiParserLogger = LoggerFactory.getLogger(IJsonSmiParser.class);

    default ArrayList<Accelerator> jsonOutputToAccelerators(
            JsonElement rootObject, LinkedHashSet<Integer> parsedAcceleratorIds) {

        ArrayList<Accelerator> accelerators = new ArrayList<>();
        List<JsonObject> acceleratorObjects = extractAccelerators(rootObject);

        for (JsonObject acceleratorObject : acceleratorObjects) {
            Integer acceleratorId = extractAcceleratorId(acceleratorObject);
            if (acceleratorId != null
                    && (parsedAcceleratorIds.isEmpty()
                            || parsedAcceleratorIds.contains(acceleratorId))) {
                Accelerator accelerator = jsonObjectToAccelerator(acceleratorObject);
                accelerators.add(accelerator);
            }
        }

        return accelerators;
    }

    public Integer extractAcceleratorId(JsonObject jsonObject);

    public Accelerator jsonObjectToAccelerator(JsonObject jsonObject);

    public List<JsonObject> extractAccelerators(JsonElement rootObject);
}
