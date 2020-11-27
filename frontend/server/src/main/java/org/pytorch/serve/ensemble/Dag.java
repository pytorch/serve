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

    public ArrayList<String> topoSort() throws InvalidDAGException {

        Map<String, Integer> inDegreeMap = getInDegreeMap();
        ArrayList<String> topoSortedList = new ArrayList<String>();
        Set<String> startNodes = getStartNodeNames();
        Set<String> leafNodes = getLeafNodeNames();

        if (startNodes.size() != 1) {
            throw new InvalidDAGException("DAG should have only one start node");
        }

        if (leafNodes.size() != 1) {
            throw new InvalidDAGException("DAG should have only one end node");
        }

        Set<String> zeroInDegree = startNodes;
        Set<String> executing = new HashSet<>();

        for (String s : zeroInDegree) {
            nodes.get(s).updateInputDataMap("start", "0");
        }

        while (!zeroInDegree.isEmpty()) {
            Set<String> readyToExecute = new HashSet<>(zeroInDegree);
            readyToExecute.removeAll(executing);
            executing.addAll(readyToExecute);

            ArrayList<NodeOutput> outputs = execute(readyToExecute);

            for (NodeOutput output : outputs) {
                String nodeName = output.getNodeName();
                executing.remove(nodeName);
                zeroInDegree.remove(nodeName);
                topoSortedList.add(nodeName);

                for (String newNodeName : dagMap.get(nodeName).get("outDegree")) {
                    nodes.get(newNodeName).updateInputDataMap(nodeName, output.getData());
                    inDegreeMap.replace(newNodeName, inDegreeMap.get(newNodeName) - 1);
                    if (inDegreeMap.get(newNodeName) == 0) {
                        zeroInDegree.add(newNodeName);
                    }
                }
            }
        }

        if (topoSortedList.size() != nodes.size()) {
            throw new InvalidDAGException("Not a valid DAG");
        }

        executorService.shutdown();
        return topoSortedList;
    }

    private ArrayList<NodeOutput> execute(Set<String> readyToExecute) {
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
}
