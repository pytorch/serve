package org.pytorch.serve.device.utils;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import org.pytorch.serve.device.Accelerator;
import org.pytorch.serve.device.AcceleratorVendor;
import org.pytorch.serve.device.interfaces.IAcceleratorUtility;
import org.pytorch.serve.device.interfaces.IJsonSmiParser;

public class RblnUtil implements IAcceleratorUtility, IJsonSmiParser {

    @Override
    public String getGpuEnvVariableName() {
        return "RBLN_DEVICES";
    }

    @Override
    public String[] getUtilizationSmiCommand() {
        return new String[] {
            "rbln-stat", "-j"
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
        String model = gpuObject.get("name").getAsString();
        if (!model.startsWith("RBLN")) {
            return null;
        }
        int npuId = gpuObject.get("npu").getAsInt();
        float npuUtil = gpuObject.get("util").getAsFloat();
        long memoryTotal = gpuObject.getAsJsonObject("memory").get("total").getAsLong();
        long memoryUsed = gpuObject.getAsJsonObject("memory").get("used").getAsLong();

        Accelerator accelerator = new Accelerator(model, AcceleratorVendor.RBLN, npuId);

        // Set additional information
        accelerator.setUsagePercentage(npuUtil);
        accelerator.setMemoryUtilizationPercentage((memoryUsed==0)?0f:(memoryUsed/(float)memoryTotal));
        accelerator.setMemoryUtilizationMegabytes((int)(memoryUsed/1024/1024));

        return accelerator;
    }

    @Override
    public Integer extractAcceleratorId(JsonObject cardObject) {
        Integer npuId = cardObject.get("npu").getAsInt();
        return npuId;
    }

    @Override
    public List<JsonObject> extractAccelerators(JsonElement rootObject) {
        List<JsonObject> accelerators = new ArrayList<>();
        JsonArray devicesArray =
            rootObject
            .getAsJsonObject()
            .get("devices")
            .getAsJsonArray();

        for (JsonElement elem : devicesArray){
            accelerators.add(elem.getAsJsonObject());
        }

        return accelerators;
    }

    public ArrayList<Accelerator> jsonOutputToAccelerators(
            JsonObject rootObject, LinkedHashSet<Integer> parsedAcceleratorIds) {

        ArrayList<Accelerator> accelerators = new ArrayList<>();
        List<JsonObject> acceleratorObjects = extractAccelerators(rootObject);

        int i=0;
        for (JsonObject acceleratorObject : acceleratorObjects) {
            Integer acceleratorId = extractAcceleratorId(acceleratorObject);
            if (acceleratorId != null
                    && (parsedAcceleratorIds.isEmpty()
                        || parsedAcceleratorIds.contains(acceleratorId))) {
                Accelerator accelerator = jsonObjectToAccelerator(acceleratorObject);
                accelerators.add(accelerator);
            }
            i++;
        }

        return accelerators;
    }
}
