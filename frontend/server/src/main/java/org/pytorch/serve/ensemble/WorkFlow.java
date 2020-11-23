package org.pytorch.serve.ensemble;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.ExecutionException;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.InvalidModelException;
import org.pytorch.serve.archive.model.Manifest;
import org.pytorch.serve.archive.model.ModelArchive;
import org.pytorch.serve.archive.model.ModelException;
import org.pytorch.serve.archive.utils.ZipUtils;
import org.pytorch.serve.http.InternalServerException;
import org.pytorch.serve.http.StatusResponse;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.wlm.ModelManager;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.error.YAMLException;

public class WorkFlow {
    private LinkedHashMap<String, Object> obj;
    private int minWorkers = 1;
    private int maxWorkers = 1;
    private int batchSize = 1;
    private int batchSizeDelay = 50;
    private Map<String, WorkflowModel> models;
    private Dag dag = new Dag();
    private File dir;
    public static String MANIFEST_FILE = "MANIFEST.json";
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private WorkflowManifest workflowManifest;
    private File specFile;
    private File handlerFile;

    public WorkFlow(File workFlowFile) throws Exception {
        String name = FilenameUtils.getBaseName(workFlowFile.getName());

        InputStream is = Files.newInputStream(workFlowFile.toPath());
        this.dir = ZipUtils.unzip(is, null, "models");
        this.workflowManifest = load(this.dir, true);
        this.specFile = new File(this.dir, this.workflowManifest.getWorfklow().getSpecFile());
        this.handlerFile = new File(this.dir, this.workflowManifest.getWorfklow().getHandler());
        this.models = new HashMap<String, WorkflowModel>();
        this.obj = (LinkedHashMap<String, Object>) this.readFile(this.specFile);

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
                wfm = new WorkflowModel(modelName, null, 1, 1, 1, 0, this.handlerFile.getPath());
            } else {
                wfm = models.get(modelName);
            }
            Node fromNode = new Node(modelName, wfm);
            dag.addNode(fromNode);
            for (String toModelName : (ArrayList<String>) entry.getValue()) {
                WorkflowModel toWfm;
                if (!models.containsKey(modelName)) {
                    toWfm =
                            new WorkflowModel(
                                    modelName, null, 1, 1, 1, 0, this.handlerFile.getPath());
                } else {
                    toWfm = models.get(modelName);
                }
                Node toNode = new Node(toModelName, toWfm);
                dag.addNode(toNode);
                dag.addEdge(fromNode, toNode);
            }
        }
    }

    private static Object readFile(File file) throws InvalidModelException, IOException {
        Yaml yaml = new Yaml();
        try (Reader r =
                new InputStreamReader(
                        Files.newInputStream(file.toPath()), StandardCharsets.UTF_8)) {
            return yaml.load(r);
        } catch (YAMLException e) {
            throw new InvalidModelException("Failed to parse yaml.", e);
        }
    }

    private static WorkflowManifest load(File dir, boolean extracted)
            throws InvalidModelException, IOException {
        boolean failed = true;
        try {
            File manifestFile = new File(dir, "WAR-INF/" + MANIFEST_FILE);
            WorkflowManifest manifest = null;
            if (manifestFile.exists()) {
                manifest = readFile(manifestFile, WorkflowManifest.class);
            } else {
                manifest = new WorkflowManifest();
            }

            failed = false;
            return manifest;
        } finally {
            if (extracted && failed) {
                FileUtils.deleteQuietly(dir);
            }
        }
    }

    private static <T> T readFile(File file, Class<T> type)
            throws InvalidModelException, IOException {
        try (Reader r =
                new InputStreamReader(
                        Files.newInputStream(file.toPath()), StandardCharsets.UTF_8)) {
            return GSON.fromJson(r, type);
        } catch (JsonParseException e) {
            throw new InvalidModelException("Failed to parse signature.json.", e);
        }
    }



    public Object getObj() {
        return obj;
    }
}
