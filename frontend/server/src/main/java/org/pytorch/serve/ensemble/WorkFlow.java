package org.pytorch.serve.ensemble;

import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import org.pytorch.serve.archive.workflow.InvalidWorkflowException;
import org.pytorch.serve.archive.workflow.WorkflowArchive;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.error.YAMLException;

public class WorkFlow {

    private Map<String, Object> workflowSpec;

    private WorkflowArchive workflowArchive;
    private int minWorkers = 1;
    private int maxWorkers = 1;
    private int batchSize = 1;
    private int maxBatchDelay = 50;
    private int timeOutMs = 10000;
    private int retryAttempts = 1;

    private Dag dag = new Dag();

    public WorkFlow(WorkflowArchive workflowArchive)
            throws IOException, InvalidDAGException, InvalidWorkflowException {
        this.workflowArchive = workflowArchive;
        File specFile =
                new File(
                        this.workflowArchive.getWorkflowDir(),
                        this.workflowArchive.getManifest().getWorkflow().getSpecFile());
        File handlerFile =
                new File(
                        this.workflowArchive.getWorkflowDir(),
                        this.workflowArchive.getManifest().getWorkflow().getHandler());

        String workFlowName = this.workflowArchive.getWorkflowName();
        Map<String, WorkflowModel> models = new HashMap<String, WorkflowModel>();

        @SuppressWarnings("unchecked")
        LinkedHashMap<String, Object> spec =
                (LinkedHashMap<String, Object>) this.readSpecFile(specFile);
        this.workflowSpec = spec;

        @SuppressWarnings("unchecked")
        Map<String, Object> modelsInfo = (Map<String, Object>) this.workflowSpec.get("models");
        for (Map.Entry<String, Object> entry : modelsInfo.entrySet()) {
            String keyName = entry.getKey();

            switch (keyName) {
                case "min-workers":
                    minWorkers = (int) entry.getValue();
                    break;
                case "max-workers":
                    maxWorkers = (int) entry.getValue();
                    break;
                case "batch-size":
                    batchSize = (int) entry.getValue();
                    break;
                case "max-batch-delay":
                    maxBatchDelay = (int) entry.getValue();
                    break;
                case "retry-attempts":
                    retryAttempts = (int) entry.getValue();
                    break;
                case "timeout-ms":
                    timeOutMs = (int) entry.getValue();
                    break;
                default:
                    // entry.getValue().getClass() check object type.
                    // assuming Map containing model info
                    @SuppressWarnings("unchecked")
                    LinkedHashMap<String, Object> model =
                            (LinkedHashMap<String, Object>) entry.getValue();
                    String modelName = workFlowName + "__" + keyName;

                    WorkflowModel wfm =
                            new WorkflowModel(
                                    modelName,
                                    (String) model.get("url"),
                                    (int) model.getOrDefault("min-workers", minWorkers),
                                    (int) model.getOrDefault("max-workers", maxWorkers),
                                    (int) model.getOrDefault("batch-size", batchSize),
                                    (int) model.getOrDefault("max-batch-delay", maxBatchDelay),
                                    (int) model.getOrDefault("retry-attempts", retryAttempts),
                                    (int) model.getOrDefault("timeout-ms", timeOutMs),
                                    null);

                    models.put(modelName, wfm);
            }
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> dagInfo = (Map<String, Object>) this.workflowSpec.get("dag");

        for (Map.Entry<String, Object> entry : dagInfo.entrySet()) {
            String nodeName = entry.getKey();
            String modelName = workFlowName + "__" + nodeName;
            WorkflowModel wfm;
            if (!models.containsKey(modelName)) {
                wfm =
                        new WorkflowModel(
                                modelName,
                                null,
                                1,
                                1,
                                1,
                                0,
                                retryAttempts,
                                timeOutMs,
                                handlerFile.getPath() + ":" + nodeName);
            } else {
                wfm = models.get(modelName);
            }
            Node fromNode = new Node(nodeName, wfm);
            dag.addNode(fromNode);

            @SuppressWarnings("unchecked")
            ArrayList<String> values = (ArrayList<String>) entry.getValue();
            for (String toNodeName : values) {

                if (toNodeName == null || ("").equals(toNodeName.strip())) {
                    continue;
                }
                String toModelName = workFlowName + "__" + toNodeName;
                WorkflowModel toWfm;
                if (!models.containsKey(toModelName)) {
                    toWfm =
                            new WorkflowModel(
                                    toModelName,
                                    null,
                                    1,
                                    1,
                                    1,
                                    0,
                                    retryAttempts,
                                    timeOutMs,
                                    handlerFile.getPath() + ":" + toNodeName);
                } else {
                    toWfm = models.get(toModelName);
                }
                Node toNode = new Node(toNodeName, toWfm);
                dag.addNode(toNode);
                dag.addEdge(fromNode, toNode);
            }
        }
        dag.validate();
    }

    private static Map<String, Object> readSpecFile(File file)
            throws IOException, InvalidWorkflowException {
        Yaml yaml = new Yaml();
        try (Reader r =
                new InputStreamReader(
                        Files.newInputStream(file.toPath()), StandardCharsets.UTF_8)) {
            @SuppressWarnings("unchecked")
            Map<String, Object> loadedYaml = (Map<String, Object>) yaml.load(r);
            return loadedYaml;
        } catch (YAMLException e) {
            throw new InvalidWorkflowException("Failed to parse yaml.", e);
        }
    }

    public Object getWorkflowSpec() {
        return workflowSpec;
    }

    public Dag getDag() {
        return this.dag;
    }

    public WorkflowArchive getWorkflowArchive() {
        return workflowArchive;
    }

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

    public String getWorkflowDag() {
        return this.workflowSpec.get("dag").toString();
    }
}
