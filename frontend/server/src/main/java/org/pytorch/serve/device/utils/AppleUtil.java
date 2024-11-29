package org.pytorch.serve.device.utils;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.pytorch.serve.device.Accelerator;
import org.pytorch.serve.device.AcceleratorVendor;
import org.pytorch.serve.device.interfaces.IAcceleratorUtility;
import org.pytorch.serve.device.interfaces.IJsonSmiParser;

public class AppleUtil implements IAcceleratorUtility, IJsonSmiParser {

    @Override
    public String getGpuEnvVariableName() {
        return null; // Apple doesn't use a GPU environment variable
    }

    @Override
    public String[] getUtilizationSmiCommand() {
        return new String[] {
            "system_profiler", "-json", "-detailLevel", "mini", "SPDisplaysDataType"
        };
    }

    @Override
    public ArrayList<Accelerator> getAvailableAccelerators(
            LinkedHashSet<Integer> availableAcceleratorIds) {
        String jsonOutput = IAcceleratorUtility.callSMI(getUtilizationSmiCommand());
        JsonObject rootObject = JsonParser.parseString(jsonOutput).getAsJsonObject();
        return jsonOutputToAccelerators(rootObject, availableAcceleratorIds);
    }

    @Override
    public ArrayList<Accelerator> smiOutputToUpdatedAccelerators(
            String smiOutput, LinkedHashSet<Integer> parsedGpuIds) {
        JsonObject rootObject = JsonParser.parseString(smiOutput).getAsJsonObject();
        return jsonOutputToAccelerators(rootObject, parsedGpuIds);
    }

    @Override
    public Accelerator jsonObjectToAccelerator(JsonObject gpuObject) {
        String model = gpuObject.get("sppci_model").getAsString();
        if (!model.startsWith("Apple M")) {
            return null;
        }

        Accelerator accelerator = new Accelerator(model, AcceleratorVendor.APPLE, 0);

        // Set additional information
        accelerator.setUsagePercentage(0f); // Not available from system_profiler
        accelerator.setMemoryUtilizationPercentage(0f); // Not available from system_profiler
        accelerator.setMemoryUtilizationMegabytes(0); // Not available from system_profiler

        return accelerator;
    }

    @Override
    public Integer extractAcceleratorId(JsonObject cardObject) {
        // `system_profiler` only returns one object for
        // the integrated GPU on M1, M2, M3 Macs
        return 0;
    }

    @Override
    public List<JsonObject> extractAccelerators(JsonElement rootObject) {
        List<JsonObject> accelerators = new ArrayList<>();
        JsonArray displaysArray =
                rootObject
                        .getAsJsonObject() // Gets the outer object
                        .get("SPDisplaysDataType") // Gets the "SPDisplaysDataType" element
                        .getAsJsonArray();
        JsonObject gpuObject = displaysArray.get(0).getAsJsonObject();
        int number_of_cores = Integer.parseInt(gpuObject.get("sppci_cores").getAsString());

        // add the object `number_of_cores` times to maintain the exsisitng
        // functionality
        accelerators =
                IntStream.range(0, number_of_cores)
                        .mapToObj(i -> gpuObject)
                        .collect(Collectors.toList());

        return accelerators;
    }

    public ArrayList<Accelerator> jsonOutputToAccelerators(
            JsonObject rootObject, LinkedHashSet<Integer> parsedAcceleratorIds) {

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
}
