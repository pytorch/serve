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
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.pytorch.serve.archive.model.ModelNotFoundException;
import org.pytorch.serve.archive.model.ModelVersionNotFoundException;
import org.pytorch.serve.http.InternalServerException;
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

    public ArrayList<NodeOutput> execute(RequestInput input, ArrayList<String> topoSortedList) {

        CompletionService<NodeOutput> executorCompletionService = null;
        if (topoSortedList == null) {
            ExecutorService executorService = Executors.newFixedThreadPool(4);
            executorCompletionService = new ExecutorCompletionService<>(executorService);
        }

        Map<String, Integer> inDegreeMap = this.dag.getInDegreeMap();
        Set<String> zeroInDegree = dag.getStartNodeNames();
        Set<String> executing = new HashSet<>();

        if (topoSortedList == null) {
            for (String s : zeroInDegree) {
                RequestInput newInput = new RequestInput(UUID.randomUUID().toString());
                newInput.setHeaders(input.getHeaders());
                newInput.setParameters(input.getParameters());
                inputRequestMap.put(s, newInput);
            }
        }

        ArrayList<NodeOutput> leafOutputs = new ArrayList<>();

        while (!zeroInDegree.isEmpty()) {
            Set<String> readyToExecute = new HashSet<>(zeroInDegree);
            readyToExecute.removeAll(executing);
            executing.addAll(readyToExecute);

            ArrayList<NodeOutput> outputs = new ArrayList<>();
            if (topoSortedList == null) {
                for (String name : readyToExecute) {
                    executorCompletionService.submit(
                            () ->
                                    invokeModel(
                                            name,
                                            this.dag.getNodes().get(name).getWorkflowModel(),
                                            inputRequestMap.get(name),
                                            0));
                }

                try {
                    Future<NodeOutput> op = executorCompletionService.take();
                    if (op == null) {
                        throw new ExecutionException(
                                new RuntimeException("WorkflowNode result empty"));
                    } else {
                        outputs.add(op.get());
                    }
                } catch (InterruptedException | ExecutionException e) {
                    logger.error(e.getMessage());
                    String[] error = e.getMessage().split(":");
                    throw new InternalServerException(error[error.length - 1]); // NOPMD
                }
            } else {
                for (String name : readyToExecute) {
                    outputs.add(new NodeOutput(name, null));
                }
            }

            for (NodeOutput output : outputs) {
                String nodeName = output.getNodeName();
                executing.remove(nodeName);
                zeroInDegree.remove(nodeName);

                if (topoSortedList != null) {
                    topoSortedList.add(nodeName);
                }

                Set<String> childNodes = this.dag.getDagMap().get(nodeName).get("outDegree");
                if (childNodes.isEmpty()) {
                    leafOutputs.add(output);
                } else {
                    for (String newNodeName : childNodes) {

                        if (topoSortedList == null) {
                            byte[] response = (byte[]) output.getData();

                            RequestInput newInput = this.inputRequestMap.get(newNodeName);
                            if (newInput == null) {
                                List<InputParameter> params = new ArrayList<>();
                                newInput = new RequestInput(UUID.randomUUID().toString());
                                if (inDegreeMap.get(newNodeName) == 1) {
                                    params.add(new InputParameter("body", response));
                                } else {
                                    params.add(new InputParameter(nodeName, response));
                                }
                                newInput.setParameters(params);
                                newInput.setHeaders(input.getHeaders());
                            } else {
                                newInput.addParameter(new InputParameter(nodeName, response));
                            }
                            this.inputRequestMap.put(newNodeName, newInput);
                        }

                        inDegreeMap.replace(newNodeName, inDegreeMap.get(newNodeName) - 1);
                        if (inDegreeMap.get(newNodeName) == 0) {
                            zeroInDegree.add(newNodeName);
                        }
                    }
                }
            }
        }

        return leafOutputs;
    }

    private NodeOutput invokeModel(
            String nodeName, WorkflowModel workflowModel, RequestInput input, int retryAttempt)
            throws ModelNotFoundException, ModelVersionNotFoundException, ExecutionException,
                    InterruptedException {
        try {

            logger.error(String.format("Invoking -  %s for attempt %d", nodeName, retryAttempt));
            CompletableFuture<byte[]> respFuture = new CompletableFuture<>();
            RestJob job = ApiUtils.addRESTInferenceJob(null, workflowModel.getName(), null, input);
            job.setResponsePromise(respFuture);
            byte[] resp = respFuture.get(workflowModel.getTimeOutMs(), TimeUnit.MILLISECONDS);
            return new NodeOutput(nodeName, resp);
        } catch (InterruptedException | ExecutionException | TimeoutException e) {
            logger.error(e.getMessage());
            if (retryAttempt < workflowModel.getRetryAttempts()) {
                logger.error(
                        String.format(
                                "Timed out while executing %s for attempt %d",
                                nodeName, retryAttempt));
                return invokeModel(nodeName, workflowModel, input, ++retryAttempt);
            } else {
                logger.error(nodeName + " : " + e.getMessage());
                throw new InternalServerException(
                        String.format(
                                "Failed to execute workflow Node after %d attempts : Error executing %s",
                                retryAttempt, nodeName)); // NOPMD
            }
        } catch (ModelNotFoundException e) {
            logger.error("Model not found.");
            logger.error(e.getMessage());
            throw e;
        } catch (ModelVersionNotFoundException e) {
            logger.error("Model version not found.");
            logger.error(e.getMessage());
            throw e;
        }
    }
}
