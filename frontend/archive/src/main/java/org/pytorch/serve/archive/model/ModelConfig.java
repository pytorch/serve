package org.pytorch.serve.archive.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
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
                            if (v instanceof Integer) {
                                modelConfig.setMinWorkers((int) v);
                            } else {
                                logger.warn("Invalid minWorkers: {}, should be integer", v);
                            }
                            break;
                        case "maxWorkers":
                            if (v instanceof Integer) {
                                modelConfig.setMaxWorkers((int) v);
                            } else {
                                logger.warn("Invalid maxWorkers: {}, should be integer", v);
                            }
                            break;
                        case "batchSize":
                            if (v instanceof Integer) {
                                modelConfig.setBatchSize((int) v);
                            } else {
                                logger.warn("Invalid batchSize: {}, should be integer", v);
                            }
                            break;
                        case "maxBatchDelay":
                            if (v instanceof Integer) {
                                modelConfig.setMaxBatchDelay((int) v);
                            } else {
                                logger.warn("Invalid maxBatchDelay: {}, should be integer", v);
                            }
                            break;
                        case "responseTimeout":
                            if (v instanceof Integer) {
                                modelConfig.setResponseTimeout((int) v);
                            } else {
                                logger.warn("Invalid responseTimeout: {}, should be integer", v);
                            }
                            break;
                        case "deviceType":
                            if (v instanceof String) {
                                modelConfig.setDeviceType((String) v);
                            } else {
                                logger.warn("Invalid deviceType: {}, should be cpu, or gpu", v);
                            }
                            break;
                        case "parallelLevel":
                            if (v instanceof Integer) {
                                modelConfig.setParallelLevel((int) v);
                            } else {
                                logger.warn("Invalid parallelLevel: {}, should be integer >= 1", v);
                            }
                            break;
                        case "parallelType":
                            if (v instanceof String) {
                                modelConfig.setParallelMode((String) v);
                            } else {
                                logger.warn(
                                        "Invalid parallelType: {}, should be pp, tp,or pptp", v);
                            }
                            break;
                        case "deviceIds":
                            if (v instanceof List<?>) {
                                modelConfig.setDeviceIds((List<?>) v);
                            } else {
                                logger.warn("Invalid deviceIds: {}, should be list of integer", v);
                            }
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

    public void setDeviceIds(List<?> deviceIds) {
        this.deviceIds = new ArrayList<>();
        for (int i = 0; i < deviceIds.size(); i++) {
            if (deviceIds.get(i) instanceof Integer) {
                this.deviceIds.add((int) deviceIds.get(i));
            } else {
                logger.warn("Invalid deviceIds:{},", deviceIds.get(i));
                this.deviceIds = null;
                break;
            }
        }
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
        this.parallelType = ParallelType.get(parallelMode);
    }

    public ParallelType getParallelType() {
        return this.parallelType;
    }

    public void setDeviceType(String deviceType) {
        this.deviceType = DeviceType.get(deviceType);
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
            this.type = type.toLowerCase();
        }

        public String getParallelType() {
            return type;
        }

        public static ParallelType get(String parallelType) {
            ParallelType pType = NONE;
            try {
                pType =
                        Arrays.stream(ParallelType.values())
                                .filter(t -> t.type.equals(parallelType.toLowerCase()))
                                .findFirst()
                                .get();
            } catch (NoSuchElementException e) {
                logger.warn("Invalid ParallelType:{}", parallelType, e);
            }
            return pType;
        }
    }

    public enum DeviceType {
        NONE(""),
        CPU("cpu"),
        GPU("gpu");

        private String type;

        DeviceType(String type) {
            this.type = type.toLowerCase();
        }

        public String getDeviceType() {
            return type;
        }

        public static DeviceType get(String deviceType) {
            DeviceType dType = DeviceType.NONE;
            try {
                dType =
                        Arrays.stream(DeviceType.values())
                                .filter(t -> t.type.equals(deviceType.toLowerCase()))
                                .findFirst()
                                .get();
            } catch (NoSuchElementException e) {
                logger.warn("Invalid DeviceType:{}", deviceType, e);
            }
            return dType;
        }
    }
}
