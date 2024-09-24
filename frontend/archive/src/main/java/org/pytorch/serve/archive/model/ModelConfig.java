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

    public static final int defaultStartupTimeout = 120; // unit: sec
    public static final int defaultResponseTimeout = 120; // unit: sec

    /** the minimum number of workers of a model */
    private int minWorkers;
    /** the maximum number of workers of a model */
    private int maxWorkers;
    /** the batch size of a model */
    private int batchSize;
    /** the maximum delay in msec of a batch of a model */
    private int maxBatchDelay;
    /** the timeout in sec of a specific model's response. */
    private int responseTimeout = defaultResponseTimeout;
    /** the timeout in sec of a specific model's startup. */
    private int startupTimeout = defaultStartupTimeout;
    /**
     * the device type where the model is loaded. It can be gpu, cpu. The model is loaded on CPU if
     * deviceType: "cpu" is set on a GPU host.
     */
    private DeviceType deviceType = DeviceType.NONE;
    /**
     * the user specified gpu device id, By default, TorchServe auto round-robin all available GPUs
     * to assign deviceIds to a worker of a model if deviceIds is not set.
     */
    private List<Integer> deviceIds;
    /** this variable is auto calculated based on torchrun nproc-per-node. */
    private int parallelLevel;
    /** the model parallel type can be tp, pp, pptp */
    private ParallelType parallelType = ParallelType.NONE;
    /** torchrun config */
    private TorchRun torchRun;
    /** the maximum seconds of a worker recovery's timeout. default: 5 min */
    private int maxRetryTimeoutInSec = 300;
    /**
     * the client timeout in milliseconds. The inference request will be dropped once it is timeout.
     * default: 0 which means no timeout (ie. clientExpireTS default value Long.MAX_VALUE.
     */
    private long clientTimeoutInMills;
    /**
     * the job queue size of a model. By default, job_queue_size is set as 100 in config.property
     * for all models. Here, jobQueueSize: -1 means no customized setting for the model.
     */
    private int jobQueueSize;
    /**
     * the useJobTicket is a flag which allows an inference request to be accepted only if there are
     * available workers.
     */
    private boolean useJobTicket;
    /**
     * the max idle in milliseconds of a sequence inference request of this stateful model. The
     * default value is 0.
     */
    private long sequenceMaxIdleMSec;
    /**
     * the timeout of a sequence inference request of this stateful model. The default value is 0.
     */
    private long sequenceTimeoutMSec;
    /**
     * the job queue size of one inference sequence of this stateful model. The default value is 1.
     */
    private int maxSequenceJobQueueSize = 1;
    /** the max number of sequences can be accepted. The default value is 1. */
    private int maxNumSequence = 1;
    /** continuousBatching is a flag to enable continuous batching. */
    private boolean continuousBatching;
    /** asyncCommunication is a flag to enable async communication. */
    private boolean asyncCommunication;
    /**
     * Create python virtual environment when using python backend to install model dependencies (if
     * enabled globally using configuration install_py_dep_per_model=true) and run workers for model
     * loading and inference.
     */
    private boolean useVenv;
    /** sequenceBatching is a flag to enable https://github.com/pytorch/serve/issues/2743 */
    private boolean sequenceBatching;

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
                        case "startupTimeout":
                            if (v instanceof Integer) {
                                modelConfig.setStartupTimeout((int) v);
                            } else {
                                logger.warn("Invalid startupTimeout: {}, should be integer", v);
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
                        case "parallelLevel":
                            if (v instanceof Integer) {
                                modelConfig.setParallelLevel((int) v);
                            } else {
                                logger.warn("Invalid parallelLevel: {}, should be integer", v);
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
                        case "maxRetryTimeoutInSec":
                            if (v instanceof Integer) {
                                modelConfig.setMaxRetryTimeoutInSec((int) v);
                            } else {
                                logger.warn(
                                        "Invalid maxRetryTimeoutInMin: {}, should be integer", v);
                            }
                            break;
                        case "clientTimeoutInMills":
                            if (v instanceof Integer) {
                                modelConfig.setClientTimeoutInMills(((Integer) v).longValue());
                            } else {
                                logger.warn(
                                        "Invalid clientTimeoutInMills: {}, should be positive long",
                                        v);
                            }
                            break;
                        case "jobQueueSize":
                            if (v instanceof Integer) {
                                modelConfig.setJobQueueSize((int) v);
                            } else {
                                logger.warn("Invalid jobQueueSize: {}, should be positive int", v);
                            }
                            break;
                        case "useJobTicket":
                            if (v instanceof Boolean) {
                                modelConfig.setUseJobTicket((boolean) v);
                            } else {
                                logger.warn("Invalid useJobTicket: {}, should be true or false", v);
                            }
                            break;
                        case "sequenceMaxIdleMSec":
                            if (v instanceof Integer) {
                                modelConfig.setSequenceMaxIdleMSec(((Integer) v).longValue());
                            } else {
                                logger.warn(
                                        "Invalid sequenceMaxIdleMSec: {}, should be positive int",
                                        v);
                            }
                            break;
                        case "sequenceTimeoutMSec":
                            if (v instanceof Integer) {
                                modelConfig.setSequenceTimeoutMSec(((Integer) v).longValue());
                            } else {
                                logger.warn(
                                        "Invalid sequenceTimeoutMSec: {}, should be positive int",
                                        v);
                            }
                            break;
                        case "maxSequenceJobQueueSize":
                            if (v instanceof Integer) {
                                modelConfig.setMaxSequenceJobQueueSize((int) v);
                            } else {
                                logger.warn(
                                        "Invalid maxSequenceJobQueueSize: {}, should be positive int",
                                        v);
                            }
                            break;
                        case "maxNumSequence":
                            if (v instanceof Integer) {
                                modelConfig.setMaxNumSequence((int) v);
                            } else {
                                logger.warn(
                                        "Invalid maxNumSequence: {}, should be positive int", v);
                            }
                            break;
                        case "continuousBatching":
                            if (v instanceof Boolean) {
                                modelConfig.setContinuousBatching((boolean) v);
                            } else {
                                logger.warn(
                                        "Invalid continuousBatching: {}, should be true or false",
                                        v);
                            }
                            break;
                        case "sequenceBatching":
                            if (v instanceof Boolean) {
                                modelConfig.setSequenceBatching((boolean) v);
                            } else {
                                logger.warn(
                                        "Invalid sequenceBatching: {}, should be true or false", v);
                            }
                            break;
                        case "asyncCommunication":
                            if (v instanceof Boolean) {
                                modelConfig.setAsyncCommunication((boolean) v);
                            } else {
                                logger.warn(
                                        "Invalid asyncCommunication: {}, should be true or false",
                                        v);
                            }
                            break;
                        case "useVenv":
                            if (v instanceof Boolean) {
                                modelConfig.setUseVenv((boolean) v);
                            } else {
                                logger.warn("Invalid useVenv: {}, should be true or false", v);
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
        this.minWorkers = Math.max(1, minWorkers);
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
        this.batchSize = Math.max(1, batchSize);
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

    public int getStartupTimeout() {
        return startupTimeout;
    }

    public void setStartupTimeout(int startupTimeout) {
        if (startupTimeout <= 0) {
            logger.warn("Invalid startupTimeout:{}", startupTimeout);
            return;
        }
        this.startupTimeout = startupTimeout;
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
        if (parallelLevel < 0) {
            logger.warn("Invalid parallelLevel:{}, set as 0", parallelLevel);
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

    public int getMaxRetryTimeoutInSec() {
        return maxRetryTimeoutInSec;
    }

    public void setMaxRetryTimeoutInSec(int maxRetryTimeoutInSec) {
        this.maxRetryTimeoutInSec = Math.max(0, maxRetryTimeoutInSec);
    }

    public long getClientTimeoutInMills() {
        return clientTimeoutInMills;
    }

    public void setClientTimeoutInMills(long clientTimeoutInMills) {
        this.clientTimeoutInMills = Math.max(0, clientTimeoutInMills);
    }

    public int getJobQueueSize() {
        return jobQueueSize;
    }

    public void setJobQueueSize(int jobQueueSize) {
        this.jobQueueSize = Math.max(0, jobQueueSize);
    }

    public boolean isUseJobTicket() {
        return useJobTicket;
    }

    public void setUseJobTicket(boolean useJobTicket) {
        this.useJobTicket = useJobTicket;
    }

    public long getSequenceMaxIdleMSec() {
        return sequenceMaxIdleMSec;
    }

    public void setSequenceMaxIdleMSec(long sequenceMaxIdleMSec) {
        this.sequenceMaxIdleMSec = Math.max(0, sequenceMaxIdleMSec);
    }

    public long getSequenceTimeoutMSec() {
        return sequenceTimeoutMSec;
    }

    public void setSequenceTimeoutMSec(long sequenceTimeoutMSec) {
        this.sequenceTimeoutMSec = Math.max(0, sequenceTimeoutMSec);
    }

    public int getMaxSequenceJobQueueSize() {
        return maxSequenceJobQueueSize;
    }

    public void setMaxSequenceJobQueueSize(int maxsequenceJobQueueSize) {
        this.maxSequenceJobQueueSize = Math.max(1, maxsequenceJobQueueSize);
    }

    public boolean isContinuousBatching() {
        return continuousBatching;
    }

    public boolean isAsyncCommunication() {
        return asyncCommunication;
    }

    public void setContinuousBatching(boolean continuousBatching) {
        this.continuousBatching = continuousBatching;
    }

    public boolean isSequenceBatching() {
        return sequenceBatching;
    }

    public void setSequenceBatching(boolean sequenceBatching) {
        this.sequenceBatching = sequenceBatching;
    }

    public void setAsyncCommunication(boolean asyncCommunication) {
        this.asyncCommunication = asyncCommunication;
    }

    public int getMaxNumSequence() {
        return maxNumSequence;
    }

    public void setMaxNumSequence(int maxNumSequence) {
        this.maxNumSequence = Math.max(1, maxNumSequence);
    }

    public boolean getUseVenv() {
        return useVenv;
    }

    public void setUseVenv(boolean useVenv) {
        this.useVenv = useVenv;
    }

    public enum ParallelType {
        NONE(""),
        PP("pp"),
        TP("tp"),
        PPTP("pptp"),
        CUSTOM("custom");

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
        private int monitorInterval = 5;
        private int nodeRank;
        private String masterAddr;
        private int masterPort;
        private int ompNumberThreads = 1;

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
                            case "OMP_NUMBER_THREADS":
                                if (v instanceof Integer) {
                                    torchRun.setOmpNumberThreads((Integer) v);
                                } else {
                                    logger.warn("Invalid OMP_NUMBER_THREADS:{}, reset to 1", v);
                                }
                                break;
                            default:
                                logger.warn("unsupported parameter {}", k);
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

        public int getOmpNumberThreads() {
            return ompNumberThreads;
        }

        public void setOmpNumberThreads(int ompNumberThreads) {
            if (ompNumberThreads < 1) {
                logger.warn("Invalid OMP_NUMBER_THREADS:{}, reset to 1", ompNumberThreads);
                return;
            }
            this.ompNumberThreads = ompNumberThreads;
        }
    }
}
