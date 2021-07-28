package org.pytorch.serve.ensemble;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Direct acyclic graph for ensemble */
public class Dag {

    private Map<String, Node> nodes = new HashMap<>();

    private Map<String, Map<String, Set<String>>> dagMap = new HashMap<>();

    public void addNode(Node node) {
        if (!checkNodeExist(node)) {
            nodes.put(node.getName(), node);
            Map<String, Set<String>> degreeMap = new HashMap<>();
            degreeMap.put("inDegree", new HashSet<String>());
            degreeMap.put("outDegree", new HashSet<String>());
            dagMap.put(node.getName(), degreeMap);
        }
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

    public Map<String, Map<String, Set<String>>> getDagMap() {
        return dagMap;
    }

    public ArrayList<String> validate() throws InvalidDAGException {
        Set<String> startNodes = getStartNodeNames();

        if (startNodes.size() != 1) {
            throw new InvalidDAGException("DAG should have only one start node");
        }

        ArrayList<String> topoSortedList = new ArrayList<>();
        DagExecutor de = new DagExecutor(this);
        de.execute(null, topoSortedList);
        if (topoSortedList.size() != nodes.size()) {
            throw new InvalidDAGException("Not a valid DAG");
        }
        return topoSortedList;
    }
}
