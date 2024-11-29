package org.pytorch.serve.device;

import org.testng.Assert;
import org.testng.annotations.Test;

public class AcceleratorTest {

    @Test
    public void testAcceleratorConstructor() {
        Accelerator accelerator = new Accelerator("TestGPU", AcceleratorVendor.NVIDIA, 0);
        Assert.assertEquals(accelerator.getAcceleratorModel(), "TestGPU");
        Assert.assertEquals(accelerator.getVendor(), AcceleratorVendor.NVIDIA);
        Assert.assertEquals(accelerator.getAcceleratorId(), Integer.valueOf(0));
    }

    @Test
    public void testGettersAndSetters() {
        Accelerator accelerator = new Accelerator("TestGPU", AcceleratorVendor.AMD, 1);

        accelerator.setMemoryAvailableMegaBytes(8192);
        Assert.assertEquals(accelerator.getMemoryAvailableMegaBytes(), Integer.valueOf(8192));

        accelerator.setUsagePercentage(75.5f);
        Assert.assertEquals(accelerator.getUsagePercentage(), Float.valueOf(75.5f));

        accelerator.setMemoryUtilizationPercentage(60.0f);
        Assert.assertEquals(accelerator.getMemoryUtilizationPercentage(), Float.valueOf(60.0f));

        accelerator.setMemoryUtilizationMegabytes(4096);
        Assert.assertEquals(accelerator.getMemoryUtilizationMegabytes(), Integer.valueOf(4096));
    }

    @Test
    public void testUtilizationToString() {
        Accelerator accelerator = new Accelerator("TestGPU", AcceleratorVendor.NVIDIA, 2);
        accelerator.setUsagePercentage(80.0f);
        accelerator.setMemoryUtilizationPercentage(70.0f);
        accelerator.setMemoryUtilizationMegabytes(5120);

        String expected =
                "gpuId::2 utilization.gpu::80 % utilization.memory::70 % memory.used::5,120 MiB";
        Assert.assertEquals(accelerator.utilizationToString(), expected);
    }

    @Test
    public void testUpdateDynamicAttributes() {
        Accelerator accelerator = new Accelerator("TestGPU", AcceleratorVendor.INTEL, 3);
        accelerator.setUsagePercentage(42.42f);
        accelerator.setMemoryUtilizationPercentage(1.0f);
        accelerator.setMemoryUtilizationMegabytes(9999999);
        Accelerator updated = new Accelerator("UpdatedGPU", AcceleratorVendor.INTEL, 3);
        updated.setUsagePercentage(90.0f);
        updated.setMemoryUtilizationPercentage(85.0f);
        updated.setMemoryUtilizationMegabytes(6144);

        accelerator.updateDynamicAttributes(updated);

        Assert.assertEquals(accelerator.getUsagePercentage(), Float.valueOf(90.0f));
        Assert.assertEquals(accelerator.getMemoryUtilizationPercentage(), Float.valueOf(85.0f));
        Assert.assertEquals(accelerator.getMemoryUtilizationMegabytes(), Integer.valueOf(6144));

        // Check that static attributes are not updated
        Assert.assertEquals(accelerator.getAcceleratorModel(), "TestGPU");
        Assert.assertEquals(accelerator.getVendor(), AcceleratorVendor.INTEL);
        Assert.assertEquals(accelerator.getAcceleratorId(), Integer.valueOf(3));
    }

    @Test
    public void testAcceleratorVendorEnumValues() {
        Assert.assertEquals(AcceleratorVendor.AMD.name(), "AMD");
        Assert.assertEquals(AcceleratorVendor.NVIDIA.name(), "NVIDIA");
        Assert.assertEquals(AcceleratorVendor.INTEL.name(), "INTEL");
        Assert.assertEquals(AcceleratorVendor.APPLE.name(), "APPLE");
        Assert.assertEquals(AcceleratorVendor.UNKNOWN.name(), "UNKNOWN");
    }
}
