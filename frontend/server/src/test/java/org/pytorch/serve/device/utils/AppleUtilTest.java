package org.pytorch.serve.device.utils;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertEqualsNoOrder;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import org.pytorch.serve.device.Accelerator;
import org.pytorch.serve.device.AcceleratorVendor;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class AppleUtilTest {

    private AppleUtil appleUtil;
    private String jsonStringPath;
    private JsonObject sampleOutputJson;

    @BeforeClass
    public void setUp() {
        appleUtil = new AppleUtil();
        jsonStringPath = "src/test/resources/metrics/sample_apple_smi.json";

        try {
            FileReader reader = new FileReader(jsonStringPath);
            JsonElement jsonElement = JsonParser.parseReader(reader);
            sampleOutputJson = jsonElement.getAsJsonObject();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testGetGpuEnvVariableName() {
        assertNull(appleUtil.getGpuEnvVariableName());
    }

    @Test
    public void testGetUtilizationSmiCommand() {
        String[] expectedCommand = {
            "system_profiler", "-json", "-detailLevel", "mini", "SPDisplaysDataType"
        };
        assertEqualsNoOrder(appleUtil.getUtilizationSmiCommand(), expectedCommand);
    }

    @Test
    public void testJsonObjectToAccelerator() {
        JsonObject gpuObject =
                sampleOutputJson.getAsJsonArray("SPDisplaysDataType").get(0).getAsJsonObject();
        Accelerator accelerator = appleUtil.jsonObjectToAccelerator(gpuObject);

        assertNotNull(accelerator);
        assertEquals(accelerator.getAcceleratorModel(), "Apple M1");
        assertEquals(accelerator.getVendor(), AcceleratorVendor.APPLE);
        assertEquals(accelerator.getAcceleratorId(), Integer.valueOf(0));
        assertEquals(accelerator.getUsagePercentage(), Float.valueOf(0f));
        assertEquals(accelerator.getMemoryUtilizationPercentage(), Float.valueOf(0f));
        assertEquals(accelerator.getMemoryUtilizationMegabytes(), Integer.valueOf(0));
    }

    @Test
    public void testExtractAcceleratorId() {
        JsonObject gpuObject =
                sampleOutputJson.getAsJsonArray("SPDisplaysDataType").get(0).getAsJsonObject();
        assertEquals(appleUtil.extractAcceleratorId(gpuObject), Integer.valueOf(0));
    }

    @Test
    public void testExtractAccelerators() {
        List<JsonObject> accelerators = appleUtil.extractAccelerators(sampleOutputJson);

        assertEquals(accelerators.size(), 7);
        assertEquals(accelerators.get(0).get("sppci_model").getAsString(), "Apple M1");
    }

    @Test
    public void testSmiOutputToUpdatedAccelerators() {
        LinkedHashSet<Integer> parsedGpuIds = new LinkedHashSet<>();
        parsedGpuIds.add(0);

        ArrayList<Accelerator> updatedAccelerators =
                appleUtil.smiOutputToUpdatedAccelerators(sampleOutputJson.toString(), parsedGpuIds);

        assertEquals(updatedAccelerators.size(), 7);
        Accelerator accelerator = updatedAccelerators.get(0);
        assertEquals(accelerator.getAcceleratorModel(), "Apple M1");
        assertEquals(accelerator.getVendor(), AcceleratorVendor.APPLE);
        assertEquals(accelerator.getAcceleratorId(), Integer.valueOf(0));
    }

    @Test
    public void testGetAvailableAccelerators() {
        LinkedHashSet<Integer> availableAcceleratorIds = new LinkedHashSet<>();
        availableAcceleratorIds.add(0);

        // Mock the callSMI method to return our sample output
        AppleUtil spyAppleUtil =
                new AppleUtil() {
                    @Override
                    public String[] getUtilizationSmiCommand() {
                        return new String[] {"echo", sampleOutputJson.toString()};
                    }
                };

        ArrayList<Accelerator> availableAccelerators =
                spyAppleUtil.getAvailableAccelerators(availableAcceleratorIds);

        assertEquals(availableAccelerators.size(), 7);
        Accelerator accelerator = availableAccelerators.get(0);
        assertEquals(accelerator.getAcceleratorModel(), "Apple M1");
        assertEquals(accelerator.getVendor(), AcceleratorVendor.APPLE);
        assertEquals(accelerator.getAcceleratorId(), Integer.valueOf(0));
    }
}
