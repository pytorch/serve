package org.pytorch.serve.device.utils;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertEqualsNoOrder;

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

public class ROCmUtilTest {

    private ROCmUtil rocmUtil;
    private String sampleDiscoveryJsonPath;
    private String sampleMetricsJsonPath;
    private String sampleUpdatedMetricsJsonPath;
    private JsonObject sampleDiscoveryJsonObject;
    private JsonObject sampleMetricsJsonObject;
    private JsonObject sampleUpdatedMetricsJsonObject;

    @BeforeClass
    public void setUp() {
        rocmUtil = new ROCmUtil();
        sampleDiscoveryJsonPath = "src/test/resources/metrics/sample_amd_discovery.json";
        sampleMetricsJsonPath = "src/test/resources/metrics/sample_amd_metrics.json";
        sampleUpdatedMetricsJsonPath = "src/test/resources/metrics/sample_amd_updated_metrics.json";

        try {
            FileReader reader = new FileReader(sampleDiscoveryJsonPath);
            JsonElement jsonElement = JsonParser.parseReader(reader);
            sampleDiscoveryJsonObject = jsonElement.getAsJsonObject();

            reader = new FileReader(sampleMetricsJsonPath);
            jsonElement = JsonParser.parseReader(reader);
            sampleMetricsJsonObject = jsonElement.getAsJsonObject();

            reader = new FileReader(sampleUpdatedMetricsJsonPath);
            jsonElement = JsonParser.parseReader(reader);
            sampleUpdatedMetricsJsonObject = jsonElement.getAsJsonObject();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testGetGpuEnvVariableName() {
        assertEquals(rocmUtil.getGpuEnvVariableName(), "HIP_VISIBLE_DEVICES");
    }

    @Test
    public void testGetUtilizationSmiCommand() {
        String[] expectedCommand = {
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
        assertEqualsNoOrder(rocmUtil.getUtilizationSmiCommand(), expectedCommand);
    }

    @Test
    public void testExtractAccelerators() {
        List<JsonObject> accelerators = rocmUtil.extractAccelerators(sampleMetricsJsonObject);
        assertEquals(accelerators.size(), 2);
        assertEquals(accelerators.get(0).get("cardId").getAsString(), "card0");
        assertEquals(accelerators.get(1).get("cardId").getAsString(), "card1");
    }

    @Test
    public void testExtractAcceleratorId() {
        JsonObject card0Object = rocmUtil.extractAccelerators(sampleMetricsJsonObject).get(0);
        JsonObject card1Object = rocmUtil.extractAccelerators(sampleMetricsJsonObject).get(1);

        Integer acceleratorId0 = rocmUtil.extractAcceleratorId(card0Object);
        Integer acceleratorId1 = rocmUtil.extractAcceleratorId(card1Object);

        assertEquals(acceleratorId0, Integer.valueOf(0));
        assertEquals(acceleratorId1, Integer.valueOf(1));
    }

    @Test
    public void testJsonMetricsObjectToAccelerator() {
        JsonObject card0Object = rocmUtil.extractAccelerators(sampleMetricsJsonObject).get(0);
        Accelerator accelerator = rocmUtil.jsonObjectToAccelerator(card0Object);

        assertEquals(accelerator.getAcceleratorId(), Integer.valueOf(0));
        assertEquals(accelerator.getAcceleratorModel(), "AMD INSTINCT MI250 (MCM) OAM AC MBA");
        assertEquals(accelerator.getVendor(), AcceleratorVendor.AMD);
        assertEquals((float) accelerator.getUsagePercentage(), 50.0f);
        assertEquals((float) accelerator.getMemoryUtilizationPercentage(), 75.0f);
        assertEquals(accelerator.getMemoryAvailableMegaBytes(), Integer.valueOf(65520));
        assertEquals(accelerator.getMemoryUtilizationMegabytes(), Integer.valueOf(49140));
    }

    @Test
    public void testJsonDiscoveryObjectToAccelerator() {
        JsonObject card0Object = rocmUtil.extractAccelerators(sampleDiscoveryJsonObject).get(0);
        Accelerator accelerator = rocmUtil.jsonObjectToAccelerator(card0Object);

        assertEquals(accelerator.getAcceleratorId(), Integer.valueOf(0));
        assertEquals(accelerator.getAcceleratorModel(), "AMD INSTINCT MI250 (MCM) OAM AC MBA");
        assertEquals(accelerator.getVendor(), AcceleratorVendor.AMD);
    }

    @Test
    public void testSmiOutputToUpdatedAccelerators() {
        String smiOutput = sampleMetricsJsonObject.toString();
        String updatedMetrics = sampleUpdatedMetricsJsonObject.toString();
        LinkedHashSet<Integer> parsedGpuIds = new LinkedHashSet<>();
        parsedGpuIds.add(0);
        parsedGpuIds.add(1);

        ArrayList<Accelerator> accelerators =
                rocmUtil.smiOutputToUpdatedAccelerators(smiOutput, parsedGpuIds);
        accelerators = rocmUtil.smiOutputToUpdatedAccelerators(updatedMetrics, parsedGpuIds);

        assertEquals(accelerators.size(), 2);

        System.out.println(accelerators.toString());

        Accelerator accelerator0 = accelerators.get(0);
        assertEquals(accelerator0.getAcceleratorId(), Integer.valueOf(0));
        assertEquals(accelerator0.getAcceleratorModel(), "AMD INSTINCT MI250 (MCM) OAM AC MBA");
        assertEquals(accelerator0.getVendor(), AcceleratorVendor.AMD);
        assertEquals((float) accelerator0.getUsagePercentage(), 25.0f);
        assertEquals((float) accelerator0.getMemoryUtilizationPercentage(), 25.0f);
        assertEquals(accelerator0.getMemoryAvailableMegaBytes(), Integer.valueOf(65520));
        assertEquals(accelerator0.getMemoryUtilizationMegabytes(), Integer.valueOf(49140));
    }
}
