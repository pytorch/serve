package org.pytorch.serve.ensemble;

import io.netty.handler.codec.http.FullHttpResponse;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.messages.RequestInput;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class Node implements Callable<NodeOutput> {
    private String name;
    private String parentName;
    private Map<String, Object> inputDataMap;
    private WorkflowModel workflowModel;
    private String wfModelVersion;

    public Node(String name, WorkflowModel model) {
        this.name = name;
        this.workflowModel = model;
        this.inputDataMap = new HashMap<>();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getParentName() {
        return parentName;
    }

    public void setParentName(String parentName) {
        this.parentName = parentName;
    }

    public WorkflowModel getWorkflowModel() {
        return workflowModel;
    }

    public void setWorkflowModel(WorkflowModel workflowModel) {
        this.workflowModel = workflowModel;
    }

    public Map<String, Object> getInputDataMap() {
        return inputDataMap;
    }

    public void updateInputDataMap(String s, Object inputData) {
        this.inputDataMap.put(s, inputData);
    }

    @Override
    public NodeOutput call() throws Exception {
        return invoke_model();
//        Random rand = new Random();
//        ArrayList<String> a = new ArrayList<>();
//        for (Object s : this.getInputDataMap().values()) {
//            a.add((String) s);
//        }
//        a.add(rand.nextInt() + "");
//        return new NodeOutput(this.getName(), String.join("-", a));
    }

    private NodeOutput invoke_model(){
        try {
            //TODO remove hard coding for model version
            CompletableFuture<FullHttpResponse> respFuture = new CompletableFuture<>();
            RestJob job = ApiUtils.addInferenceJob(null, workflowModel.getName(), "1.0", (RequestInput) inputDataMap.get("input"));
            job.setResponsePromise(respFuture);
            try {
                FullHttpResponse resp = respFuture.get();
                return new NodeOutput(this.getName(), resp);
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        } catch (ModelNotFoundException e) {
            e.printStackTrace();
        } catch (ModelVersionNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }
}
