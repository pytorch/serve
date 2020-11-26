package org.pytorch.serve.ensemble;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import org.pytorch.serve.archive.model.InvalidModelException;
import org.pytorch.serve.archive.workflow.WorkflowArchive;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.error.YAMLException;

public class WorkFlow {

    private static final Logger logger = LoggerFactory.getLogger(WorkFlow.class);

    private LinkedHashMap<String, Object> obj;

    private WorkflowArchive workflowArchive;
    private int minWorkers = 1;
    private int maxWorkers = 1;
    private int batchSize = 1;
    private int batchSizeDelay = 50;

    private Map<String, WorkflowModel> models;
    private Dag dag = new Dag();
    private File dir;
    public static String MANIFEST_FILE = "MANIFEST.json";
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    // private WorkflowManifest workflowManifest;
    private File specFile;
    private File handlerFile;

    public WorkFlow(WorkflowArchive workflowArchive) throws InvalidModelException, IOException {
        this.workflowArchive = workflowArchive;
        this.specFile =
                new File(
                        this.workflowArchive.getWorkflowDir(),
                        this.workflowArchive.getManifest().getWorkflow().getSpecFile());
        this.handlerFile =
                new File(
                        this.workflowArchive.getWorkflowDir(),
                        this.workflowArchive.getManifest().getWorkflow().getHandler());
        this.models = new HashMap<String, WorkflowModel>();
        this.obj = (LinkedHashMap<String, Object>) this.readSpecFile(this.specFile);

        LinkedHashMap<String, Object> modelsInfo =
                (LinkedHashMap<String, Object>) this.obj.get("models");
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

        LinkedHashMap<String, Object> dagInfo = (LinkedHashMap<String, Object>) this.obj.get("dag");

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
            for (String toModelName : (ArrayList<String>) entry.getValue()) {
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

    private static Object readSpecFile(File file) throws InvalidModelException, IOException {
        Yaml yaml = new Yaml();
        try (Reader r =
                new InputStreamReader(
                        Files.newInputStream(file.toPath()), StandardCharsets.UTF_8)) {
            return yaml.load(r);
        } catch (YAMLException e) {
            throw new InvalidModelException("Failed to parse yaml.", e);
        }
    }

    public Object getObj() {
        return obj;
    }

    public Dag getDag() {
        return this.dag;
    }

    public WorkflowArchive getWorkflowArchive() {
        return workflowArchive;
    }
}
