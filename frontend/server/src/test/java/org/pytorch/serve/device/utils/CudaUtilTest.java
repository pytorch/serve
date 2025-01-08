package org.pytorch.serve.device.utils;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import org.pytorch.serve.device.Accelerator;
import org.pytorch.serve.device.AcceleratorVendor;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CudaUtilTest {

    private CudaUtil cudaUtil = new CudaUtil();

    @Test
    public void testGetGpuEnvVariableName() {
        Assert.assertEquals(cudaUtil.getGpuEnvVariableName(), "CUDA_VISIBLE_DEVICES");
    }

    @Test
    public void testGetUtilizationSmiCommand() {
        String[] expectedCommand = {
            "nvidia-smi",
            "--query-gpu=index,gpu_name,utilization.gpu,utilization.memory,memory.used",
            "--format=csv,nounits"
        };
        Assert.assertEquals(cudaUtil.getUtilizationSmiCommand(), expectedCommand);
    }

    @Test
    public void testSmiOutputToUpdatedAccelerators() {
        String smiOutput =
                "index,gpu_name,utilization.gpu,utilization.memory,memory.used\n"
                        + "0,NVIDIA GeForce RTX 3080,50,60,8000\n"
                        + "1,NVIDIA Tesla V100,75,80,16000\n";
        LinkedHashSet<Integer> parsedGpuIds = new LinkedHashSet<>(java.util.Arrays.asList(0, 1));

        ArrayList<Accelerator> accelerators =
                cudaUtil.smiOutputToUpdatedAccelerators(smiOutput, parsedGpuIds);

        Assert.assertEquals(accelerators.size(), 2);

        Accelerator accelerator1 = accelerators.get(0);
        Assert.assertEquals((int) accelerator1.getAcceleratorId(), 0);
        Assert.assertEquals(accelerator1.getAcceleratorModel(), "NVIDIA GeForce RTX 3080");
        Assert.assertEquals((float) accelerator1.getUsagePercentage(), 50f);
        Assert.assertEquals((float) accelerator1.getMemoryUtilizationPercentage(), 60f);
        Assert.assertEquals((int) accelerator1.getMemoryUtilizationMegabytes(), 8000);

        Accelerator accelerator2 = accelerators.get(1);
        Assert.assertEquals((int) accelerator2.getAcceleratorId(), 1);
        Assert.assertEquals(accelerator2.getAcceleratorModel(), "NVIDIA Tesla V100");
        Assert.assertEquals((float) accelerator2.getUsagePercentage(), 75f);
        Assert.assertEquals((float) accelerator2.getMemoryUtilizationPercentage(), 80f);
        Assert.assertEquals((int) accelerator2.getMemoryUtilizationMegabytes(), 16000);
    }

    @Test
    public void testParseAccelerator() {
        String[] parts = {"0", "NVIDIA GeForce RTX 3080"};
        Accelerator accelerator = cudaUtil.parseAccelerator(parts);

        Assert.assertEquals((int) accelerator.getAcceleratorId(), 0);
        Assert.assertEquals(accelerator.getAcceleratorModel(), "NVIDIA GeForce RTX 3080");
        Assert.assertEquals(accelerator.getVendor(), AcceleratorVendor.NVIDIA);
    }

    @Test
    public void testParseAcceleratorWithDifferentId() {
        String[] parts = {"2", "NVIDIA Tesla T4"};
        Accelerator accelerator = cudaUtil.parseAccelerator(parts);

        Assert.assertEquals((int) accelerator.getAcceleratorId(), 2);
        Assert.assertEquals(accelerator.getAcceleratorModel(), "NVIDIA Tesla T4");
        Assert.assertEquals(accelerator.getVendor(), AcceleratorVendor.NVIDIA);
    }

    @Test(expectedExceptions = NumberFormatException.class)
    public void testParseAcceleratorWithInvalidId() {
        String[] parts = {"invalid", "NVIDIA GeForce GTX 1080"};
        cudaUtil.parseAccelerator(parts);
    }

    @Test
    public void testParseUpdatedAccelerator() {
        String[] parts = {"1", "NVIDIA Tesla V100", "75", "80", "16000"};
        Accelerator accelerator = cudaUtil.parseUpdatedAccelerator(parts);

        Assert.assertEquals((int) accelerator.getAcceleratorId(), 1);
        Assert.assertEquals(accelerator.getAcceleratorModel(), "NVIDIA Tesla V100");
        Assert.assertEquals(accelerator.getVendor(), AcceleratorVendor.NVIDIA);
        Assert.assertEquals((float) accelerator.getUsagePercentage(), 75f);
        Assert.assertEquals((float) accelerator.getMemoryUtilizationPercentage(), 80f);
        Assert.assertEquals((int) accelerator.getMemoryUtilizationMegabytes(), 16000);
    }

    @Test
    public void testParseUpdatedAcceleratorWithDifferentValues() {
        String[] parts = {"3", "NVIDIA A100", "30.5", "45.7", "40960"};
        Accelerator accelerator = cudaUtil.parseUpdatedAccelerator(parts);

        Assert.assertEquals((int) accelerator.getAcceleratorId(), 3);
        Assert.assertEquals(accelerator.getAcceleratorModel(), "NVIDIA A100");
        Assert.assertEquals(accelerator.getVendor(), AcceleratorVendor.NVIDIA);
        Assert.assertEquals((float) accelerator.getUsagePercentage(), 30.5f);
        Assert.assertEquals((float) accelerator.getMemoryUtilizationPercentage(), 45.7f);
        Assert.assertEquals((int) accelerator.getMemoryUtilizationMegabytes(), 40960);
    }

    @Test(expectedExceptions = NumberFormatException.class)
    public void testParseUpdatedAcceleratorWithInvalidUsagePercentage() {
        String[] parts = {"0", "NVIDIA GeForce RTX 2080", "invalid", "80", "8000"};
        cudaUtil.parseUpdatedAccelerator(parts);
    }

    @Test(expectedExceptions = NumberFormatException.class)
    public void testParseUpdatedAcceleratorWithInvalidMemoryUtilization() {
        String[] parts = {"0", "NVIDIA GeForce RTX 2080", "75", "invalid", "8000"};
        cudaUtil.parseUpdatedAccelerator(parts);
    }

    @Test(expectedExceptions = NumberFormatException.class)
    public void testParseUpdatedAcceleratorWithInvalidMemoryUsage() {
        String[] parts = {"0", "NVIDIA GeForce RTX 2080", "75", "80", "invalid"};
        cudaUtil.parseUpdatedAccelerator(parts);
    }

    @Test(expectedExceptions = ArrayIndexOutOfBoundsException.class)
    public void testParseUpdatedAcceleratorWithInsufficientData() {
        String[] parts = {"0", "NVIDIA GeForce RTX 2080", "75", "80"};
        cudaUtil.parseUpdatedAccelerator(parts);
    }
}
