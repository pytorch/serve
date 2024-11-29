package org.pytorch.serve.device;

import java.text.MessageFormat;
import org.pytorch.serve.device.interfaces.IAcceleratorUtility;

public class Accelerator {
    public final Integer id;
    public final AcceleratorVendor vendor;
    public final String model;
    public IAcceleratorUtility acceleratorUtility;
    public Float usagePercentage;
    public Float memoryUtilizationPercentage;
    public Integer memoryAvailableMegabytes;
    public Integer memoryUtilizationMegabytes;

    public Accelerator(String acceleratorName, AcceleratorVendor vendor, Integer gpuId) {
        this.model = acceleratorName;
        this.vendor = vendor;
        this.id = gpuId;
        this.usagePercentage = (float) 0.0;
        this.memoryUtilizationPercentage = (float) 0.0;
        this.memoryAvailableMegabytes = 0;
        this.memoryUtilizationMegabytes = 0;
    }

    // Getters
    public Integer getMemoryAvailableMegaBytes() {
        return memoryAvailableMegabytes;
    }

    public AcceleratorVendor getVendor() {
        return vendor;
    }

    public String getAcceleratorModel() {
        return model;
    }

    public Integer getAcceleratorId() {
        return id;
    }

    public Float getUsagePercentage() {
        return usagePercentage;
    }

    public Float getMemoryUtilizationPercentage() {
        return memoryUtilizationPercentage;
    }

    public Integer getMemoryUtilizationMegabytes() {
        return memoryUtilizationMegabytes;
    }

    // Setters
    public void setMemoryAvailableMegaBytes(Integer memoryAvailable) {
        this.memoryAvailableMegabytes = memoryAvailable;
    }

    public void setUsagePercentage(Float acceleratorUtilization) {
        this.usagePercentage = acceleratorUtilization;
    }

    public void setMemoryUtilizationPercentage(Float memoryUtilizationPercentage) {
        this.memoryUtilizationPercentage = memoryUtilizationPercentage;
    }

    public void setMemoryUtilizationMegabytes(Integer memoryUtilizationMegabytes) {
        this.memoryUtilizationMegabytes = memoryUtilizationMegabytes;
    }

    // Other Methods
    public String utilizationToString() {
        final String message =
                MessageFormat.format(
                        "gpuId::{0} utilization.gpu::{1} % utilization.memory::{2} % memory.used::{3} MiB",
                        id,
                        usagePercentage,
                        memoryUtilizationPercentage,
                        memoryUtilizationMegabytes);

        return message;
    }

    public void updateDynamicAttributes(Accelerator updated) {
        this.usagePercentage = updated.usagePercentage;
        this.memoryUtilizationPercentage = updated.memoryUtilizationPercentage;
        this.memoryUtilizationMegabytes = updated.memoryUtilizationMegabytes;
    }
}
