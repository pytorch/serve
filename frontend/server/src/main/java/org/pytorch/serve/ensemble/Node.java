package org.pytorch.serve.ensemble;

import io.netty.handler.codec.http.FullHttpResponse;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.messages.RequestInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Node implements Callable<NodeOutput> {

    private static final Logger logger = LoggerFactory.getLogger(Node.class);
    private String name;
    private String parentName;
    private Map<String, Object> inputDataMap;
    private WorkflowModel workflowModel;

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
        return invokeModel();
        //        Random rand = new Random();
        //        ArrayList<String> a = new ArrayList<>();
        //        for (Object s : this.getInputDataMap().values()) {
        //            a.add((String) s);
        //        }
        //        a.add(rand.nextInt() + "");
        //        return new NodeOutput(this.getName(), String.join("-", a));
    }

    private NodeOutput invokeModel() {
        try {
            // TODO remove hard coding for model version
            CompletableFuture<FullHttpResponse> respFuture = new CompletableFuture<>();
            RestJob job =
                    ApiUtils.addInferenceJob(
                            null,
                            workflowModel.getName(),
                            null,
                            (RequestInput) inputDataMap.get("input"));
            job.setResponsePromise(respFuture);
            try {
                FullHttpResponse resp = respFuture.get();

                return new NodeOutput(this.getName(), resp);
            } catch (InterruptedException | ExecutionException e) {
                logger.error("Failed to execute workflow Node.");
                logger.error(e.getMessage());
            }
        } catch (ModelNotFoundException | ModelVersionNotFoundException e) {
            logger.error("Model not found.");
            logger.error(e.getMessage());
        }
        return null;
    }
}
