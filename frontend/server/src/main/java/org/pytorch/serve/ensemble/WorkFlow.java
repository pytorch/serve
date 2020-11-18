package org.pytorch.serve.ensemble;

import com.google.gson.JsonParseException;
import org.pytorch.serve.archive.InvalidModelException;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.error.YAMLException;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;

import org.apache.commons.io.FilenameUtils;


public class WorkFlow {
    private LinkedHashMap<String, Object> obj;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private Map<String, WorkflowModel> models;
    private Dag dag;


    public WorkFlow(String path) throws Exception {
        File workFlowFile = new File(path);
        String name = FilenameUtils.getBaseName(workFlowFile.getName());
        this.obj = (LinkedHashMap<String, Object>) this.readFile(workFlowFile);

        LinkedHashMap<String, Object> modelsInfo = (LinkedHashMap<String, Object>) this.obj.get("models");
        for (  Map.Entry<String,Object> entry : modelsInfo.entrySet() ) {
                String keyName = entry.getKey();

                switch(keyName){
                    case "min-workers":
                        minWorkers = (int) entry.getValue();
                    case "max-workers":
                        maxWorkers = (int) entry.getValue();
                    case "batch-size":
                        batchSize = (int) entry.getValue();
                    default:
                        // entry.getValue().getClass() check object type.
                        // assuming Map containing model info

                        LinkedHashMap<String, Object> model = (LinkedHashMap<String, Object>) entry.getValue();

                        WorkflowModel wfm = new WorkflowModel(keyName,
                                (String) model.get("url"),
                                (int) model.get("min-workers"),
                                (int) model.get("max-workers"),
                                (int) model.get("batch-size"));

                        models.put(keyName, wfm);
                }

        }

        LinkedHashMap<String, Object> dagInfo = (LinkedHashMap<String, Object>) this.obj.get("dag");

        for ( Map.Entry<String,Object> entry : dagInfo.entrySet()) {
            String modelName = entry.getKey();
            WorkflowModel wfm;
            if(!models.containsKey(modelName)){
                 wfm  = new WorkflowModel(modelName,
                        "handler.py",
                        1,
                       1,
                        -1);
            }else{
                wfm =  models.get(modelName);
            }
            Node fromNode = new Node(modelName, wfm);
            dag.addNode(fromNode);
            for(String toModelName: (ArrayList<String>)entry.getValue()){
                WorkflowModel toWfm;
                if(!models.containsKey(modelName)){
                    toWfm  = new WorkflowModel(modelName,
                            "handler.py",
                            1,
                            1,
                            -1);
                }else{
                    toWfm =  models.get(modelName);
                }
                Node toNode = new Node(toModelName, toWfm);
                dag.addNode(toNode);
                dag.addEdge(fromNode, toNode);
            }
        }
    }

    private static Object readFile(File file)
            throws InvalidModelException, IOException {
        Yaml yaml = new Yaml();
        try (Reader r =
                     new InputStreamReader(
                             Files.newInputStream(file.toPath()), StandardCharsets.UTF_8)) {
            return yaml.load(r);
        } catch (YAMLException e) {
            throw new InvalidModelException("Failed to parse yaml.", e);
        }
    }

//    private static Object register() {
//         //obj.get('dag');
//
//
//    }


    public Object getObj() {
        return obj;
    }


}



