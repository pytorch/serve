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

    private LinkedHashMap<String, Object> workflowSpec;

    private WorkflowArchive workflowArchive;
    private int minWorkers = 1;
    private int maxWorkers = 1;
    private int batchSize = 1;
    private int batchSizeDelay = 50;

    private Map<String, WorkflowModel> models;
    private Dag dag = new Dag();
    private File handlerFile;

    public WorkFlow(WorkflowArchive workflowArchive)
            throws IOException, InvalidDAGException, InvalidWorkflowException {
        this.workflowArchive = workflowArchive;
        File specFile =
                new File(
                        this.workflowArchive.getWorkflowDir(),
                        this.workflowArchive.getManifest().getWorkflow().getSpecFile());
        this.handlerFile =
                new File(
                        this.workflowArchive.getWorkflowDir(),
                        this.workflowArchive.getManifest().getWorkflow().getHandler());
        this.models = new HashMap<String, WorkflowModel>();
        @SuppressWarnings("unchecked")
        LinkedHashMap<String, Object> spec = (LinkedHashMap<String, Object>) this.readSpecFile(specFile);
        this.workflowSpec = spec;

        @SuppressWarnings("unchecked")
        LinkedHashMap<String, Object> modelsInfo =
                (LinkedHashMap<String, Object>) this.workflowSpec.get("models");
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
                case "batch-size-delay":
                    batchSizeDelay = (int) entry.getValue();
                    break;
                default:
                    // entry.getValue().getClass() check object type.
                    // assuming Map containing model info
                    @SuppressWarnings("unchecked")
                    LinkedHashMap<String, Object> model =
                            (LinkedHashMap<String, Object>) entry.getValue();

                    WorkflowModel wfm =
                            new WorkflowModel(
                                    keyName,
                                    (String) model.get("url"),
                                    (int) model.getOrDefault("min-workers", minWorkers),
                                    (int) model.getOrDefault("max-workers", maxWorkers),
                                    (int) model.getOrDefault("batch-size", batchSize),
                                    (int) model.getOrDefault("batch-size-delay", batchSizeDelay),
                                    null);

                    models.put(keyName, wfm);
            }
        }

        @SuppressWarnings("unchecked")
        LinkedHashMap<String, Object> dagInfo =
                (LinkedHashMap<String, Object>) this.workflowSpec.get("dag");

        for (Map.Entry<String, Object> entry : dagInfo.entrySet()) {
            String modelName = entry.getKey();
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
                                this.handlerFile.getPath() + ":" + modelName);
            } else {
                wfm = models.get(modelName);
            }
            Node fromNode = new Node(modelName, wfm);
            dag.addNode(fromNode);

            @SuppressWarnings("unchecked")
            ArrayList<String> values = (ArrayList<String>) entry.getValue();
            for (String toModelName : values ) {
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
                                    this.handlerFile.getPath() + ":" + toModelName);
                } else {
                    toWfm = models.get(toModelName);
                }
                Node toNode = new Node(toModelName, toWfm);
                dag.addNode(toNode);
                dag.addEdge(fromNode, toNode);
            }
        }
    }

    private static Object readSpecFile(File file) throws IOException, InvalidWorkflowException {
        Yaml yaml = new Yaml();
        try (Reader r =
                new InputStreamReader(
                        Files.newInputStream(file.toPath()), StandardCharsets.UTF_8)) {
            return yaml.load(r);
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

    public int getBatchSizeDelay() {
        return batchSizeDelay;
    }

    public void setBatchSizeDelay(int batchSizeDelay) {
        this.batchSizeDelay = batchSizeDelay;
    }

    public String getWorkflowDag() {
        return this.workflowSpec.get("dag").toString();
    }
}
