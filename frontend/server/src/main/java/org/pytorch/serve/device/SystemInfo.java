package org.pytorch.serve.device;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.pytorch.serve.device.interfaces.IAcceleratorUtility;
import org.pytorch.serve.device.utils.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SystemInfo {
    static final Logger logger = LoggerFactory.getLogger(SystemInfo.class);
    //
    // Contains information about the system (physical or virtual machine)
    // we are running the workload on.
    // Specifically how many accelerators and info about them.
    //

    public AcceleratorVendor acceleratorVendor;
    ArrayList<Accelerator> accelerators;
    private IAcceleratorUtility acceleratorUtil;

    public SystemInfo() {
        // Detect and set the vendor of any accelerators in the system
        this.acceleratorVendor = detectVendorType();
        this.accelerators = new ArrayList<Accelerator>();

        // If accelerators are present (vendor != UNKNOWN),
        // initialize accelerator utilities
        Optional.of(hasAccelerators())
                // Only proceed if hasAccelerators() returns true
                .filter(Boolean::booleanValue)
                // Execute this block if accelerators are present
                .ifPresent(
                        hasAcc -> {
                            // Create the appropriate utility class based on vendor
                            this.acceleratorUtil = createAcceleratorUtility();
                            // Populate the accelerators list based on environment
                            // variables and available devices
                            populateAccelerators();
                        });

        // Safely handle accelerator metrics update
        Optional.ofNullable(accelerators)
                // Only proceed if the accelerators list is not empty
                .filter(list -> !list.isEmpty())
                // Update metrics (utilization, memory, etc.) for all accelerators if list
                // exists and not empty
                .ifPresent(list -> updateAcceleratorMetrics());
    }

    private IAcceleratorUtility createAcceleratorUtility() {
        switch (this.acceleratorVendor) {
            case AMD:
                return new ROCmUtil();
            case NVIDIA:
                return new CudaUtil();
            case INTEL:
                return new XpuUtil();
            case APPLE:
                return new AppleUtil();
            default:
                return null;
        }
    }

    private void populateAccelerators() {
        if (this.acceleratorUtil != null) {
            String envVarName = this.acceleratorUtil.getGpuEnvVariableName();
            String requestedAcceleratorIds = System.getenv(envVarName);
            LinkedHashSet<Integer> availableAcceleratorIds =
                    IAcceleratorUtility.parseVisibleDevicesEnv(requestedAcceleratorIds);
            this.accelerators =
                    this.acceleratorUtil.getAvailableAccelerators(availableAcceleratorIds);
        } else {
            this.accelerators = new ArrayList<>();
        }
    }

    boolean hasAccelerators() {
        return this.acceleratorVendor != AcceleratorVendor.UNKNOWN;
    }

    public Integer getNumberOfAccelerators() {
        // since we instance create `accelerators` as an empty list
        // in the constructor, the null check should be redundant.
        // leaving it to be sure.
        return (accelerators != null) ? accelerators.size() : 0;
    }

    public static AcceleratorVendor detectVendorType() {
        if (isCommandAvailable("rocm-smi")) {
            return AcceleratorVendor.AMD;
        } else if (isCommandAvailable("nvidia-smi")) {
            return AcceleratorVendor.NVIDIA;
        } else if (isCommandAvailable("xpu-smi")) {
            return AcceleratorVendor.INTEL;
        } else if (isCommandAvailable("system_profiler")) {
            return AcceleratorVendor.APPLE;
        } else {
            return AcceleratorVendor.UNKNOWN;
        }
    }

    private static boolean isCommandAvailable(String command) {
        String operatingSystem = System.getProperty("os.name").toLowerCase();
        String commandCheck = operatingSystem.contains("win") ? "where" : "which";
        ProcessBuilder processBuilder = new ProcessBuilder(commandCheck, command);
        try {
            Process process = processBuilder.start();
            int exitCode = process.waitFor();
            return exitCode == 0;
        } catch (IOException | InterruptedException e) {
            return false;
        }
    }

    public ArrayList<Accelerator> getAccelerators() {
        return this.accelerators;
    }

    private void updateAccelerators(List<Accelerator> updatedAccelerators) {
        // Create a map of existing accelerators with ID as key
        Map<Integer, Accelerator> existingAcceleratorsMap =
                this.accelerators.stream().collect(Collectors.toMap(acc -> acc.id, acc -> acc));

        // Update existing accelerators and add new ones
        this.accelerators =
                updatedAccelerators.stream()
                        .map(
                                updatedAcc -> {
                                    Accelerator existingAcc =
                                            existingAcceleratorsMap.get(updatedAcc.id);
                                    if (existingAcc != null) {
                                        existingAcc.updateDynamicAttributes(updatedAcc);
                                        return existingAcc;
                                    } else {
                                        return updatedAcc;
                                    }
                                })
                        .collect(Collectors.toCollection(ArrayList::new));
    }

    public void updateAcceleratorMetrics() {
        if (this.acceleratorUtil != null) {
            List<Accelerator> updatedAccelerators =
                    this.acceleratorUtil.getUpdatedAcceleratorsUtilization(this.accelerators);

            updateAccelerators(updatedAccelerators);
        }
    }

    public AcceleratorVendor getAcceleratorVendor() {
        return this.acceleratorVendor;
    }

    public String getVisibleDevicesEnvName() {
        if (this.accelerators.isEmpty() || this.accelerators == null) {
            return null;
        }
        return this.accelerators.get(0).acceleratorUtility.getGpuEnvVariableName();
    }
}
