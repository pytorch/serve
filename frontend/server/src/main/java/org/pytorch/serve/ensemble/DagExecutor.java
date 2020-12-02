package org.pytorch.serve.ensemble;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.job.RestJob;
import org.pytorch.serve.util.ApiUtils;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.RequestInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DagExecutor {

    private static final Logger logger = LoggerFactory.getLogger(DagExecutor.class);

    private Dag dag;
    private Map<String, RequestInput> inputRequestMap;

    public DagExecutor(Dag dag) {
        this.dag = dag;
        inputRequestMap = new ConcurrentHashMap<>();
    }

    public ArrayList<NodeOutput> execute(RequestInput input) {

        ExecutorService executorService = Executors.newFixedThreadPool(4);
        CompletionService<NodeOutput> executorCompletionService =
                new ExecutorCompletionService<>(executorService);

        Map<String, Integer> inDegreeMap = this.dag.getInDegreeMap();

        Set<String> zeroInDegree = dag.getStartNodeNames();
        Set<String> executing = new HashSet<>();

        for (String s : zeroInDegree) {
            RequestInput newInput = new RequestInput(UUID.randomUUID().toString());
            newInput.setHeaders(input.getHeaders());
            newInput.setParameters(input.getParameters());
            inputRequestMap.put(s, newInput);
        }

        NodeOutput output = null;
        ArrayList<NodeOutput> leafOutputs = new ArrayList<>();

        while (!zeroInDegree.isEmpty()) {
            Set<String> readyToExecute = new HashSet<>(zeroInDegree);
            readyToExecute.removeAll(executing);
            executing.addAll(readyToExecute);
            for (String name : readyToExecute) {
                executorCompletionService.submit(
                        () ->
                                invokeModel(
                                        name,
                                        this.dag.getNodes().get(name).getWorkflowModel(),
                                        inputRequestMap.get(name)));
            }

            try {
                output = executorCompletionService.take().get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace(); // NOPMD
            }

            String nodeName = output.getNodeName();
            executing.remove(nodeName);
            zeroInDegree.remove(nodeName);

            Set<String> childNodes = this.dag.getDagMap().get(nodeName).get("outDegree");
            if (childNodes.isEmpty()) {
                leafOutputs.add(output);
            } else {
                for (String newNodeName : childNodes) {
                    List<InputParameter> params = new ArrayList<>();
                    RequestInput newInput = new RequestInput(UUID.randomUUID().toString());
                    byte[] response = (byte[]) output.getData();
                    params.add(new InputParameter("body", response));
                    newInput.setParameters(params);
                    newInput.setHeaders(input.getHeaders());
                    this.inputRequestMap.put(newNodeName, newInput);
                    inDegreeMap.replace(newNodeName, inDegreeMap.get(newNodeName) - 1);
                    if (inDegreeMap.get(newNodeName) == 0) {
                        zeroInDegree.add(newNodeName);
                    }
                }
            }
        }

        return leafOutputs;
    }

    public NodeOutput invokeModel(
            String nodeName, WorkflowModel workflowModel, RequestInput input) {
        try {
            CompletableFuture<byte[]> respFuture = new CompletableFuture<>();

            RestJob job = ApiUtils.addRESTInferenceJob(null, workflowModel.getName(), null, input);
            job.setResponsePromise(respFuture);
            try {
                byte[] resp = respFuture.get();
                return new NodeOutput(nodeName, resp);
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
