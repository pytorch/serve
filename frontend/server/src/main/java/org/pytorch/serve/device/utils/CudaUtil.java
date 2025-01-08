package org.pytorch.serve.device.utils;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import org.pytorch.serve.device.Accelerator;
import org.pytorch.serve.device.AcceleratorVendor;
import org.pytorch.serve.device.interfaces.IAcceleratorUtility;
import org.pytorch.serve.device.interfaces.ICsvSmiParser;

public class CudaUtil implements IAcceleratorUtility, ICsvSmiParser {

    @Override
    public String getGpuEnvVariableName() {
        return "CUDA_VISIBLE_DEVICES";
    }

    @Override
    public String[] getUtilizationSmiCommand() {
        String metrics =
                String.join(
                        ",",
                        "index",
                        "gpu_name",
                        "utilization.gpu",
                        "utilization.memory",
                        "memory.used");
        return new String[] {"nvidia-smi", "--query-gpu=" + metrics, "--format=csv,nounits"};
    }

    @Override
    public ArrayList<Accelerator> getAvailableAccelerators(
            LinkedHashSet<Integer> availableAcceleratorIds) {
        String[] command = {"nvidia-smi", "--query-gpu=index,gpu_name", "--format=csv,nounits"};

        String smiOutput = IAcceleratorUtility.callSMI(command);
        return csvSmiOutputToAccelerators(
                smiOutput, availableAcceleratorIds, this::parseAccelerator);
    }

    @Override
    public ArrayList<Accelerator> smiOutputToUpdatedAccelerators(
            String smiOutput, LinkedHashSet<Integer> parsedGpuIds) {

        return csvSmiOutputToAccelerators(smiOutput, parsedGpuIds, this::parseUpdatedAccelerator);
    }

    public Accelerator parseAccelerator(String[] parts) {
        int id = Integer.parseInt(parts[0].trim());
        String model = parts[1].trim();
        return new Accelerator(model, AcceleratorVendor.NVIDIA, id);
    }

    public Accelerator parseUpdatedAccelerator(String[] parts) {
        int id = Integer.parseInt(parts[0].trim());
        String model = parts[1].trim();
        Float usagePercentage = Float.parseFloat(parts[2].trim());
        Float memoryUtilizationPercentage = Float.parseFloat(parts[3].trim());
        int memoryUtilizationMegabytes = Integer.parseInt(parts[4].trim());

        Accelerator accelerator = new Accelerator(model, AcceleratorVendor.NVIDIA, id);
        accelerator.setUsagePercentage(usagePercentage);
        accelerator.setMemoryUtilizationPercentage(memoryUtilizationPercentage);
        accelerator.setMemoryUtilizationMegabytes(memoryUtilizationMegabytes);
        return accelerator;
    }
}
