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
    private TorchRun torchRun;

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
                        case "torchrun":
                            if (v instanceof Map<?, ?>) {
                                modelConfig.torchRun = TorchRun.build((Map<?, ?>) v);
                                modelConfig.setParallelLevel(
                                        modelConfig.torchRun.getNprocPerNode());
                            } else {
                                logger.warn(
                                        "Invalid torchrun: {}, should be Torchrun parameters", v);
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

    public TorchRun getTorchRun() {
        return torchRun;
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

    public static class TorchRun {
        private int nnodes = 1;
        private int nprocPerNode = 1;
        private String rdzvId;
        private String rdzvEndpoint;
        private String rdzvBackend = "c10d";
        private String rdzvConf;
        private int maxRestarts = 3;
        private int monitorInterval = 5;
        private int nodeRank;
        private String masterAddr;
        private int masterPort;

        public static TorchRun build(Map<?, ?> torchRunMap) {
            TorchRun torchRun = new TorchRun();
            torchRunMap.forEach(
                    (k, v) -> {
                        switch ((String) k) {
                            case "nnodes":
                                if (v instanceof Integer) {
                                    torchRun.setNnodes((Integer) v);
                                } else {
                                    logger.warn("Invalid torchrun.nnodes:{}, reset to 1", v);
                                }
                                break;
                            case "nproc-per-node":
                                if (v instanceof Integer) {
                                    torchRun.setNprocPerNode((Integer) v);
                                } else {
                                    logger.warn(
                                            "Invalid torchrun.nproc-per-node:{}, reset to 1", v);
                                }
                                break;
                            case "rdzv-backend":
                                if (v instanceof String) {
                                    torchRun.setRdzvBackend((String) v);
                                } else {
                                    logger.warn(
                                            "Invalid torchrun.rdzv-backend:{}, reset to c10d", v);
                                }
                                break;
                            case "rdzv-endpoint":
                                if (v instanceof String) {
                                    torchRun.setRdzvEndpoint((String) v);
                                } else {
                                    logger.warn("Invalid torchrun.rdzv-endpoint:{}", v);
                                }
                                break;
                            case "rdzv-conf":
                                if (v instanceof String) {
                                    torchRun.setRdzvConf((String) v);
                                } else {
                                    logger.warn("Invalid torchrun.rdzv-conf:{}", v);
                                }
                                break;
                            case "max-restarts":
                                if (v instanceof Integer) {
                                    torchRun.setMaxRestarts((Integer) v);
                                } else {
                                    logger.warn("Invalid torchrun.max-restarts:{}, reset to 3", v);
                                }
                                break;
                            case "monitor-interval":
                                if (v instanceof Integer) {
                                    torchRun.setMonitorInterval((Integer) v);
                                } else {
                                    logger.warn("Invalid torchrun.max-restarts:{}, reset to 5", v);
                                }
                                break;
                            case "node-rank":
                                if (v instanceof Integer) {
                                    torchRun.setNodeRank((Integer) v);
                                } else {
                                    logger.warn("Invalid torchrun.node-rank:{}, reset to 0", v);
                                }
                                break;
                            default:
                                break;
                        }
                    });
            return torchRun;
        }

        public int getNnodes() {
            return nnodes;
        }

        public void setNnodes(int nnodes) {
            if (nnodes <= 0) {
                logger.warn("Invalid torchrun.nnodes:{}, reset to 1", nnodes);
                return;
            }
            this.nnodes = nnodes;
        }

        public int getNprocPerNode() {
            return nprocPerNode;
        }

        public void setNprocPerNode(int nprocPerNode) {
            if (nprocPerNode <= 0) {
                logger.warn("Invalid torchrun.nproc-per-node:{}, reset to 1", nprocPerNode);
                return;
            }
            this.nprocPerNode = nprocPerNode;
        }

        public String getRdzvId() {
            return rdzvId;
        }

        public void setRdzvId(String rdzvId) {
            this.rdzvId = rdzvId;
        }

        public String getRdzvEndpoint() {
            return rdzvEndpoint;
        }

        public void setRdzvEndpoint(String rdzvEndpoint) {
            this.rdzvEndpoint = rdzvEndpoint;
        }

        public String getRdzvBackend() {
            return rdzvBackend;
        }

        public void setRdzvBackend(String rdzvBackend) {
            this.rdzvBackend = rdzvBackend;
        }

        public String getRdzvConf() {
            return rdzvConf;
        }

        public void setRdzvConf(String rdzvConf) {
            this.rdzvConf = rdzvConf;
        }

        public int getMaxRestarts() {
            return maxRestarts;
        }

        public void setMaxRestarts(int maxRestarts) {
            if (maxRestarts <= 0) {
                logger.warn("Invalid torchrun.max-restarts:{}, reset to 3", maxRestarts);
                return;
            }
            this.maxRestarts = maxRestarts;
        }

        public int getMonitorInterval() {
            return monitorInterval;
        }

        public void setMonitorInterval(int monitorInterval) {
            if (monitorInterval <= 0) {
                logger.warn("Invalid torchrun.monitor-interval:{}, reset to 5", monitorInterval);
                return;
            }
            this.monitorInterval = monitorInterval;
        }

        public int getNodeRank() {
            return nodeRank;
        }

        public void setNodeRank(int nodeRank) {
            if (nodeRank < 0) {
                logger.warn("Invalid torchrun.node-rank:{}, reset to 0", nodeRank);
                return;
            }
            this.nodeRank = nodeRank;
        }

        public String getMasterAddr() {
            return masterAddr;
        }

        public void setMasterAddr(String masterAddr) {
            this.masterAddr = masterAddr;
        }

        public int getMasterPort() {
            return masterPort;
        }

        public void setMasterPort(int masterPort) {
            this.masterPort = masterPort;
        }
    }
}
