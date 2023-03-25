package org.pytorch.serve.archive.model;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelConfig {
    private static final Logger logger = LoggerFactory.getLogger(ModelConfig.class);

    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private int responseTimeout;
    private DeviceType deviceType = DeviceType.NONE;
    private List<Integer> deviceIds;
    private int parallelLevel = 1;
    private ParallelType parallelType = ParallelType.NONE;

    public static ModelConfig build(Map<String, Object> yamlMap) {
        ModelConfig modelConfig = new ModelConfig();
        yamlMap.forEach(
                (k, v) -> {
                    switch (k) {
                        case "minWorkers":
                            modelConfig.setMinWorkers((int) v);
                            break;
                        case "maxWorkers":
                            modelConfig.setMaxWorkers((int) v);
                            break;
                        case "batchSize":
                            modelConfig.setBatchSize((int) v);
                            break;
                        case "maxBatchDelay":
                            modelConfig.setMaxBatchDelay((int) v);
                            break;
                        case "responseTimeout":
                            modelConfig.setResponseTimeout((int) v);
                            break;
                        case "deviceType":
                            modelConfig.setDeviceType((String) v);
                            break;
                        case "parallelLevel":
                            modelConfig.setParallelLevel((int) v);
                            break;
                        case "parallelType":
                            modelConfig.setParallelMode((String) v);
                            break;
                        case "deviceIds":
                            modelConfig.setDeviceIds(v);
                            break;
                        default:
                            break;
                    }
                });
        return modelConfig;
    }

    public int getMinWorkers() {
        return minWorkers;
    }

    public void setMinWorkers(int minWorkers) {
        if (minWorkers < 0) {
            logger.warn("Invalid minWorkers:{}", minWorkers);
            return;
        }
        this.minWorkers = minWorkers;
    }

    public int getMaxWorkers() {
        return maxWorkers;
    }

    public void setMaxWorkers(int maxWorkers) {
        if (maxWorkers < 0) {
            logger.warn("Invalid maxWorkers:{}", maxWorkers);
            return;
        }
        this.maxWorkers = maxWorkers;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            logger.warn("Invalid batchSize:{}", batchSize);
            return;
        }
        this.batchSize = batchSize;
    }

    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    public void setMaxBatchDelay(int maxBatchDelay) {
        if (maxBatchDelay < 0) {
            logger.warn("Invalid maxBatchDelay:{}", maxBatchDelay);
            return;
        }
        this.maxBatchDelay = maxBatchDelay;
    }

    public int getResponseTimeout() {
        return responseTimeout;
    }

    public void setResponseTimeout(int responseTimeout) {
        if (responseTimeout <= 0) {
            logger.warn("Invalid responseTimeout:{}", responseTimeout);
            return;
        }
        this.responseTimeout = responseTimeout;
    }

    public List<Integer> getDeviceIds() {
        return deviceIds;
    }

    public void setDeviceIds(Object deviceIds) {
        this.deviceIds =
                Stream.of(deviceIds)
                        .map(Object::toString)
                        .map(Integer::parseInt)
                        .collect(Collectors.toList());
    }

    public int getParallelLevel() {
        return parallelLevel;
    }

    public void setParallelLevel(int parallelLevel) {
        if (parallelLevel <= 0) {
            logger.warn("Invalid parallelLevel:{}, set as 1", parallelLevel);
            this.parallelLevel = 1;
            return;
        }
        this.parallelLevel = parallelLevel;
    }

    public void setParallelMode(String parallelMode) {
        this.parallelType = ParallelType.get(parallelMode).get();
    }

    public ParallelType getParallelType() {
        return this.parallelType;
    }

    public void setDeviceType(String deviceType) {
        this.deviceType = DeviceType.get(deviceType).get();
    }

    public DeviceType getDeviceType() {
        return deviceType;
    }

    public enum ParallelType {
        NONE(""),
        PP("pp"),
        TP("tp"),
        PPTP("pptp");

        private String type;

        ParallelType(String type) {
            this.type = type;
        }

        public String getParallelType() {
            return type;
        }

        public static Optional<ParallelType> get(String parallelType) {
            return Arrays.stream(ParallelType.values())
                    .filter(t -> t.type.equals(parallelType))
                    .findFirst();
        }
    }

    public enum DeviceType {
        NONE(""),
        CPU("cpu"),
        GPU("gpu"),
        NEURON("neuron");

        private String type;

        DeviceType(String type) {
            this.type = type;
        }

        public String getDeviceType() {
            return type;
        }

        public static Optional<DeviceType> get(String deviceType) {
            return Arrays.stream(DeviceType.values())
                    .filter(t -> t.type.equals(deviceType))
                    .findFirst();
        }
    }
}
