package org.pytorch.serve.device.utils;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import org.pytorch.serve.device.Accelerator;
import org.pytorch.serve.device.AcceleratorVendor;
import org.pytorch.serve.device.interfaces.IAcceleratorUtility;
import org.pytorch.serve.device.interfaces.ICsvSmiParser;

public class XpuUtil implements IAcceleratorUtility, ICsvSmiParser {

    @Override
    public String getGpuEnvVariableName() {
        return "XPU_VISIBLE_DEVICES";
    }

    @Override
    public ArrayList<Accelerator> getAvailableAccelerators(
            final LinkedHashSet<Integer> availableAcceleratorIds) {
        final String[] smiCommand = {
            "xpu-smi",
            "discovery",
            "--dump", // output as csv
            String.join(
                    ",",
                    "1", // device Id
                    "2", // Device name
                    "16" // Memory physical size
                    )
        };
        final String smiOutput = IAcceleratorUtility.callSMI(smiCommand);

        final String acceleratorEnv = getGpuEnvVariableName();
        final String requestedAccelerators = System.getenv(acceleratorEnv);
        final LinkedHashSet<Integer> parsedAcceleratorIds =
                IAcceleratorUtility.parseVisibleDevicesEnv(requestedAccelerators);

        return csvSmiOutputToAccelerators(
                smiOutput, parsedAcceleratorIds, this::parseDiscoveryOutput);
    }

    @Override
    public final ArrayList<Accelerator> smiOutputToUpdatedAccelerators(
            final String smiOutput, final LinkedHashSet<Integer> parsedGpuIds) {
        return csvSmiOutputToAccelerators(smiOutput, parsedGpuIds, this::parseUtilizationOutput);
    }

    @Override
    public String[] getUtilizationSmiCommand() {
        // https://intel.github.io/xpumanager/smi_user_guide.html#get-the-device-real-time-statistics
        // Timestamp, DeviceId, GPU Utilization (%), GPU Memory Utilization (%)
        // 06:14:46.000, 0, 0.00, 14.61
        // 06:14:47.000, 1, 0.00, 14.59
        final String[] smiCommand = {
            "xpu-smi",
            "dump",
            "-d -1", // all devices
            "-n 1", // one dump
            "-m", // metrics
            String.join(
                    ",",
                    "0", // GPU Utilization (%), GPU active time of the elapsed time, per tile or
                    // device.
                    // Device-level is the average value of tiles for multi-tiles.
                    "5" // GPU Memory Utilization (%), per tile or device. Device-level is the
                    // average
                    // value of tiles for multi-tiles.
                    )
        };

        return smiCommand;
    }

    private Accelerator parseDiscoveryOutput(String[] parts) {
        final int acceleratorId = Integer.parseInt(parts[1].trim());
        final String deviceName = parts[2].trim();
        logger.debug("Found accelerator at index: {}, Card name: {}", acceleratorId, deviceName);
        return new Accelerator(deviceName, AcceleratorVendor.INTEL, acceleratorId);
    }

    private Accelerator parseUtilizationOutput(String[] parts) {
        final int acceleratorId = Integer.parseInt(parts[1].trim());
        final Float usagePercentage = Float.parseFloat(parts[2]);
        final Float memoryUsagePercentage = Float.parseFloat(parts[3]);
        Accelerator accelerator = new Accelerator("", AcceleratorVendor.INTEL, acceleratorId);
        accelerator.setUsagePercentage(usagePercentage);
        accelerator.setMemoryUtilizationPercentage(memoryUsagePercentage);
        return accelerator;
    }
}
