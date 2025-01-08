package org.pytorch.serve.device.utils;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.testng.Assert.*;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import org.pytorch.serve.device.Accelerator;
import org.testng.annotations.*;

public class XpuUtilTest {

    private XpuUtil xpuUtil;

    @BeforeMethod
    public void setUp() {
        xpuUtil = new XpuUtil();
    }

    @Test
    public void testGetGpuEnvVariableName() {
        assertEquals(
                xpuUtil.getGpuEnvVariableName(),
                "XPU_VISIBLE_DEVICES",
                "GPU environment variable name should be XPU_VISIBLE_DEVICES");
    }

    @Test
    public void testGetUtilizationSmiCommand() {
        String[] expectedCommand = {"xpu-smi", "dump", "-d -1", "-n 1", "-m", "0,5"};
        assertArrayEquals(
                xpuUtil.getUtilizationSmiCommand(),
                expectedCommand,
                "Utilization SMI command should match expected");
    }

    @Test
    public void testSmiOutputToUpdatedAccelerators() {
        String smiOutput =
                "Timestamp,DeviceId,GPU Utilization (%),GPU Memory Utilization (%)\n"
                        + "06:14:46.000,0,50.00,75.50\n"
                        + "06:14:47.000,1,25.00,60.25";

        LinkedHashSet<Integer> parsedGpuIds = new LinkedHashSet<>();
        parsedGpuIds.add(0);
        parsedGpuIds.add(1);

        ArrayList<Accelerator> updatedAccelerators =
                xpuUtil.smiOutputToUpdatedAccelerators(smiOutput, parsedGpuIds);

        assertEquals(updatedAccelerators.size(), 2, "Should return 2 updated accelerators");
        assertEquals(
                (int) updatedAccelerators.get(0).getAcceleratorId(),
                0,
                "First accelerator should have ID 0");
        assertEquals(
                (int) updatedAccelerators.get(1).getAcceleratorId(),
                1,
                "Second accelerator should have ID 1");
        assertEquals(
                (float) updatedAccelerators.get(0).getUsagePercentage(),
                50.00f,
                0.01,
                "GPU utilization should match for first accelerator");
        assertEquals(
                (float) updatedAccelerators.get(0).getMemoryUtilizationPercentage(),
                75.50f,
                0.01,
                "Memory utilization should match for first accelerator");
        assertEquals(
                (float) updatedAccelerators.get(1).getUsagePercentage(),
                25.00f,
                0.01,
                "GPU utilization should match for second accelerator");
        assertEquals(
                (float) updatedAccelerators.get(1).getMemoryUtilizationPercentage(),
                60.25f,
                0.01,
                "Memory utilization should match for second accelerator");
    }

    @Test
    public void testSmiOutputToUpdatedAcceleratorsWithFilteredIds() {
        String smiOutput =
                "Timestamp,DeviceId,GPU Utilization (%),GPU Memory Utilization (%)\n"
                        + "06:14:46.000,0,50.00,75.50\n"
                        + "06:14:47.000,1,25.00,60.25\n"
                        + "06:14:48.000,2,30.00,70.00";

        LinkedHashSet<Integer> parsedGpuIds = new LinkedHashSet<>();
        parsedGpuIds.add(0);
        parsedGpuIds.add(2);

        ArrayList<Accelerator> updatedAccelerators =
                xpuUtil.smiOutputToUpdatedAccelerators(smiOutput, parsedGpuIds);

        assertEquals(updatedAccelerators.size(), 2, "Should return 2 updated accelerators");
        assertEquals(
                (int) updatedAccelerators.get(0).getAcceleratorId(),
                0,
                "First accelerator should have ID 0");
        assertEquals(
                (int) updatedAccelerators.get(1).getAcceleratorId(),
                2,
                "Second accelerator should have ID 2");
        assertEquals(
                (float) updatedAccelerators.get(0).getUsagePercentage(),
                50.00f,
                0.01,
                "GPU utilization should match for first accelerator");
        assertEquals(
                (float) updatedAccelerators.get(0).getMemoryUtilizationPercentage(),
                75.50f,
                0.01,
                "Memory utilization should match for first accelerator");
        assertEquals(
                (float) updatedAccelerators.get(1).getUsagePercentage(),
                30.00f,
                0.01,
                "GPU utilization should match for third accelerator");
        assertEquals(
                (float) updatedAccelerators.get(1).getMemoryUtilizationPercentage(),
                70.00f,
                0.01,
                "Memory utilization should match for third accelerator");
    }

    @Test
    public void testSmiOutputToUpdatedAcceleratorsWithInvalidInput() {
        String invalidSmiOutput = "Invalid SMI output";
        LinkedHashSet<Integer> parsedGpuIds = new LinkedHashSet<>();
        parsedGpuIds.add(0);

        ArrayList<Accelerator> accelerators =
                xpuUtil.smiOutputToUpdatedAccelerators(invalidSmiOutput, parsedGpuIds);
        assertEquals(accelerators.size(), 0);
    }
}
