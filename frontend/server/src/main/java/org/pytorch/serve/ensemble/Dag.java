package org.pytorch.serve.ensemble;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.util.messages.RequestInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Direct acyclic graph for ensemble */
public class Dag {

    private static final Logger logger = LoggerFactory.getLogger(Dag.class);

    private Map<String, Node> nodes = new HashMap<>();
    private Map<String, Map<String, Set<String>>> dagMap = new HashMap<>();

    private ExecutorService executorService = Executors.newFixedThreadPool(4);
    private CompletionService<NodeOutput> executorCompletionService =
            new ExecutorCompletionService<>(executorService);
    private List<Future<NodeOutput>> futures = new ArrayList<Future<NodeOutput>>();

    public void addNode(Node node) {
        nodes.put(node.getName(), node);
        Map<String, Set<String>> degreeMap = new HashMap<>();
        degreeMap.put("inDegree", new HashSet<String>());
        degreeMap.put("outDegree", new HashSet<String>());
        dagMap.put(node.getName(), degreeMap);
    }

    public boolean checkNodeExist(Node node) {
        return nodes.containsKey(node.getName());
    }

    public boolean hasEdgeTo(Node from, Node to) {
        return dagMap.get(from.getName()).get("inDegree").contains(to.getName());
    }

    public void addEdge(Node from, Node to) throws InvalidDAGException {
        if (!checkNodeExist(from)) {
            addNode(from);
        }
        if (!checkNodeExist(to)) {
            addNode(to);
        }

        if (from.getName().equals(to.getName())) {
            throw new InvalidDAGException("Self loop exception");
        }

        if (hasEdgeTo(to, from)) {
            throw new InvalidDAGException("loop exception");
        }

        dagMap.get(from.getName()).get("outDegree").add(to.getName());
        dagMap.get(to.getName()).get("inDegree").add(from.getName());
    }

    public Set<String> getEndNodeNames(String degree) {
        Set<String> startNodes = new HashSet<>();
        for (Map.Entry<String, Map<String, Set<String>>> entry : dagMap.entrySet()) {
            Set<String> value = entry.getValue().get(degree);
            if (value.isEmpty()) {
                startNodes.add(entry.getKey());
            }
        }
        return startNodes;
    }

    public Set<String> getStartNodeNames() {
        return getEndNodeNames("inDegree");
    }

    public Set<String> getLeafNodeNames() {
        return getEndNodeNames("outDegree");
    }

    public Map<String, Integer> getDegreeMap(String degree) {
        Map<String, Integer> inDegreeMap = new HashMap<>();
        for (Map.Entry<String, Map<String, Set<String>>> entry : dagMap.entrySet()) {
            inDegreeMap.put(entry.getKey(), entry.getValue().get(degree).size());
        }
        return inDegreeMap;
    }

    public Map<String, Integer> getInDegreeMap() {
        return getDegreeMap("inDegree");
    }

    public Map<String, Integer> getOutDegreeMap() {
        return getDegreeMap("outDegree");
    }

    public Map<String, Node> getNodes() {
        return nodes;
    }

    public ArrayList<NodeOutput> executeFlow(RequestInput input) {
        return execute(input, null);
    }

    public ArrayList<String> topoSort() throws InvalidDAGException {
        Set<String> startNodes = getStartNodeNames();
        Set<String> leafNodes = getLeafNodeNames();

        if (startNodes.size() != 1) {
            throw new InvalidDAGException("DAG should have only one start node");
        }

        ArrayList<String> topoSortedList = new ArrayList<>();
        execute(null, topoSortedList);
        if (topoSortedList.size() != nodes.size()) {
            throw new InvalidDAGException("Not a valid DAG");
        }
        return topoSortedList;
    }

    public ArrayList<NodeOutput> execute(RequestInput input, ArrayList<String> topoSortedList) {

        Map<String, Integer> inDegreeMap = getInDegreeMap();

        Set<String> zeroInDegree = getStartNodeNames();
        Set<String> executing = new HashSet<>();

        for (String s : zeroInDegree) {
            nodes.get(s).updateInputDataMap("input", input);
        }

        ArrayList<NodeOutput> outputs = new ArrayList<>();
        ArrayList<NodeOutput> leafOutputs = new ArrayList<>();


        while (!zeroInDegree.isEmpty()) {
            Set<String> readyToExecute = new HashSet<>(zeroInDegree);
            readyToExecute.removeAll(executing);
            executing.addAll(readyToExecute);

            if (topoSortedList != null) {
                outputs = validateNode(readyToExecute);
            } else {
                outputs = executeNode(readyToExecute);
            }


            for (NodeOutput output : outputs) {
                String nodeName = output.getNodeName();
                executing.remove(nodeName);
                zeroInDegree.remove(nodeName);
                if (topoSortedList != null) {
                    topoSortedList.add(nodeName);
                }

                Set<String> outNodeNames = dagMap.get(nodeName).get("outDegree");
                if (outNodeNames.size() == 0) {
                    leafOutputs.add(output);
                } else {
                for (String newNodeName : outNodeNames) {
                    if (topoSortedList == null) {
                        List<InputParameter> params = new ArrayList<>();
                        byte[] response = (byte[]) output.getData();
                        params.add(new InputParameter("body", response));
                        input.setParameters(params);
                        nodes.get(newNodeName).updateInputDataMap("input", input);
                    }
                    inDegreeMap.replace(newNodeName, inDegreeMap.get(newNodeName) - 1);
                    if (inDegreeMap.get(newNodeName) == 0) {
                        zeroInDegree.add(newNodeName);
                    }
                }
              }
            }
        }

        executorService.shutdown();
        return leafOutputs;
    }

    public ArrayList<NodeOutput> executeNode(Set<String> readyToExecute) {
        ArrayList<NodeOutput> out = new ArrayList<>();
        for (String name : readyToExecute) {
            futures.add(executorCompletionService.submit(nodes.get(name)));
        }

        try {
            NodeOutput result = executorCompletionService.take().get();
            logger.info("Result: " + result.getNodeName() + " " + result.getData());
            out.add(result);
        } catch (InterruptedException | ExecutionException e) {
            logger.error("Failed to execute workflow Node.");
            logger.error(e.getMessage());
        }
        return out;
    }

    private ArrayList<NodeOutput> validateNode(Set<String> readyToExecute) {
        ArrayList<NodeOutput> out = new ArrayList<>();
        for (String name : readyToExecute) {
            out.add(new NodeOutput(name, null));
        }
        return out;
    }
}
