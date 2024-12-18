package org.pytorch.serve.device.utils;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.pytorch.serve.device.Accelerator;
import org.pytorch.serve.device.AcceleratorVendor;
import org.pytorch.serve.device.interfaces.IAcceleratorUtility;
import org.pytorch.serve.device.interfaces.IJsonSmiParser;

public class ROCmUtil implements IAcceleratorUtility, IJsonSmiParser {
    private static final Pattern GPU_ID_PATTERN = Pattern.compile("card(\\d+)");

    @Override
    public String getGpuEnvVariableName() {
        return "HIP_VISIBLE_DEVICES";
    }

    @Override
    public String[] getUtilizationSmiCommand() {
        return new String[] {
            "rocm-smi",
            "--showid",
            "--showproductname",
            "--showuse",
            "--showmemuse",
            "--showmeminfo",
            "vram",
            "-P",
            "--json"
        };
    }

    @Override
    public ArrayList<Accelerator> getAvailableAccelerators(
            LinkedHashSet<Integer> availableAcceleratorIds) {
        String[] smiCommand = {"rocm-smi", "--showproductname", "-P", "--json"};
        String jsonOutput = IAcceleratorUtility.callSMI(smiCommand);

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
    public List<JsonObject> extractAccelerators(JsonElement rootObject) {
        JsonObject root = rootObject.getAsJsonObject();
        List<JsonObject> accelerators = new ArrayList<>();
        for (String key : root.keySet()) {
            if (GPU_ID_PATTERN.matcher(key).matches()) {
                JsonObject accelerator = root.getAsJsonObject(key);
                accelerator.addProperty("cardId", key); // Add the card ID to the JsonObject
                accelerators.add(accelerator);
            }
        }
        return accelerators;
    }

    @Override
    public Integer extractAcceleratorId(JsonObject jsonObject) {
        String cardId = jsonObject.get("cardId").getAsString();
        Matcher matcher = GPU_ID_PATTERN.matcher(cardId);
        if (matcher.matches()) {
            return Integer.parseInt(matcher.group(1));
        }
        return null;
    }

    @Override
    public Accelerator jsonObjectToAccelerator(JsonObject jsonObject) {
        // Check if required field exists
        if (!jsonObject.has("Card Series")) {
            throw new IllegalArgumentException("Missing required field: Card Series");
        }

        String model = jsonObject.get("Card Series").getAsString();
        Integer acceleratorId = extractAcceleratorId(jsonObject);
        Accelerator accelerator = new Accelerator(model, AcceleratorVendor.AMD, acceleratorId);

        // Set optional fields using GSON's has() method
        if (jsonObject.has("GPU use (%)")) {
            accelerator.setUsagePercentage(
                    Float.parseFloat(jsonObject.get("GPU use (%)").getAsString()));
        }

        if (jsonObject.has("GPU Memory Allocated (VRAM%)")) {
            accelerator.setMemoryUtilizationPercentage(
                    Float.parseFloat(jsonObject.get("GPU Memory Allocated (VRAM%)").getAsString()));
        }

        if (jsonObject.has("VRAM Total Memory (B)")) {
            String totalMemoryStr = jsonObject.get("VRAM Total Memory (B)").getAsString().strip();
            Long totalMemoryBytes = Long.parseLong(totalMemoryStr);
            accelerator.setMemoryAvailableMegaBytes(
                    IAcceleratorUtility.bytesToMegabytes(totalMemoryBytes));
        }

        if (jsonObject.has("VRAM Total Used Memory (B)")) {
            String usedMemoryStr = jsonObject.get("VRAM Total Used Memory (B)").getAsString();
            Long usedMemoryBytes = Long.parseLong(usedMemoryStr);
            accelerator.setMemoryUtilizationMegabytes(
                    IAcceleratorUtility.bytesToMegabytes(usedMemoryBytes));
        }

        return accelerator;
    }
}
