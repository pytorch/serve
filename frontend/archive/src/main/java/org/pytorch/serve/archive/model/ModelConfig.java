package org.pytorch.serve.archive.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Optional;

public class ModelConfig {
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private int responseTimeout;
    private String deviceType;
    private CoreType coreType = CoreType.NONE;
    private ArrayList<Integer> deviceIds;
    private int parallelLevel = 1;
    private String parallelMode;
    private ParallelType parallelType = ParallelType.NONE;

    public int getMinWorkers() {
        return minWorkers;
    }

    public void setMinWorkers(int minWorkers) {
        this.minWorkers = minWorkers;
    }

    public int getMaxWorkers() {
        return maxWorkers;
    }

    public void setMaxWorkers(int maxWorkers) {
        this.maxWorkers = maxWorkers;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    public int getResponseTimeout() {
        return responseTimeout;
    }

    public void setResponseTimeout(int responseTimeout) {
        this.responseTimeout = responseTimeout;
    }

    public ArrayList<Integer> getDeviceIds() {
        return deviceIds;
    }

    public void setDeviceIds(ArrayList<Integer> deviceIds) {
        this.deviceIds = deviceIds;
    }

    public int getParallelLevel() {
        return parallelLevel;
    }

    public void setParallelLevel(int parallelLevel) {
        this.parallelLevel = parallelLevel;
    }

    public void setParallelMode(String parallelMode) {
        this.parallelMode = parallelMode;
        this.parallelType = ParallelType.get(parallelMode).get();
    }

    public String getParallelMode() {
        return this.parallelMode;
    }

    public ParallelType getParallelType() {
        return this.parallelType;
    }

    public void setDeviceType(String deviceType) {
        this.deviceType = deviceType;
        this.coreType = CoreType.get(deviceType).get();
    }

    public String getDeviceType() {
        return deviceType;
    }

    public CoreType getCoreType() {
        return coreType;
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

    public enum CoreType {
        NONE(""),
        CPU("cpu"),
        GPU("gpu"),
        NEURON("neuron");

        private String type;

        CoreType(String type) {
            this.type = type;
        }

        public String getCoreType() {
            return type;
        }

        public static Optional<CoreType> get(String coreType) {
            return Arrays.stream(CoreType.values())
                    .filter(t -> t.type.equals(coreType))
                    .findFirst();
        }
    }
}
