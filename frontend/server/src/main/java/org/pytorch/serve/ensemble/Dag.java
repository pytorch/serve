package org.pytorch.serve.ensemble;

import java.lang.reflect.Array;
import java.util.*;
import java.util.concurrent.*;

/**
 * Direct acyclic graph for ensemble
 *
 */
public class Dag {
    private Map<String, Node<?>> nodes = new HashMap<>();
    private Map<String, Map<String, Set<String>>> dagMap = new HashMap<>();

    ExecutorService executorService = Executors.newFixedThreadPool(4);
    CompletionService<NodeOutput> executorCompletionService= new ExecutorCompletionService<>(executorService);
    List<Future<Integer>> futures = new ArrayList<Future<Integer>>();

    public void addNode(Node<?> node){
        nodes.put(node.getName(), node);
        Map<String, Set<String>> degreeMap = new HashMap<>();
        degreeMap.put("inDegree", new HashSet<String>());
        degreeMap.put("outDegree", new HashSet<String>());
        dagMap.put(node.getName(), degreeMap);
    }


    public boolean isNodeExist(Node<?> node){
        return nodes.containsKey(node.getName());
    }

    public boolean hasEdgeTo(Node<?> from, Node<?> to){
        return dagMap.get(from.getName()).get("inDegree").contains(to.getName());
    }


    public void addEdge(Node<?> from, Node<?> to) throws Exception{
        if (!isNodeExist(from)){
            addNode(from);
        }
        if (!isNodeExist(to)){
            addNode(to);
        }

        if (from.getName().equals(to.getName())){
            throw new Exception("Self loop exception");
        }

        if(hasEdgeTo(to, from)){
            throw new Exception("loop exception");
        }

        dagMap.get(from.getName()).get("outDegree").add(to.getName());
        dagMap.get(to.getName()).get("inDegree").add(from.getName());

    }


    public Set<String> getEndNodeNames(String degree){
        Set<String> startNodes = new HashSet<>();
        for (Map.Entry<String, Map<String, Set<String>>>entry : dagMap.entrySet()) {
            Set<String> value = entry.getValue().get(degree);
            if(value.size() == 0) {
                startNodes.add(entry.getKey());
            }
        }
        return startNodes;
    }

    public Set<String> getStartNodeNames(){
        return getEndNodeNames("inDegree");
    }

    public Set<String> getLeafNodeNames(){
        return getEndNodeNames("outDegree");
    }

    public Map<String, Integer> getDegreeMap(String degree){
        Map<String, Integer> inDegreeMap = new HashMap<>();
        for (Map.Entry<String, Map<String, Set<String>>> entry : dagMap.entrySet()) {
                inDegreeMap.put(entry.getKey(), entry.getValue().get(degree).size());
        }
        return inDegreeMap;
    }

    public Map<String, Integer> getInDegreeMap(){
        return getDegreeMap("inDegree");
    }

    public Map<String, Integer> getOutDegreeMap(){
        return getDegreeMap("outDegree");
    }

    public Map<String, Node<?>> getNodes() {
        return nodes;
    }

    public ArrayList<String> topoSort() throws Exception {

        Map<String, Integer> inDegreeMap = getInDegreeMap();
        ArrayList<String> topoSortedList = new ArrayList<String>();
        Set<String> startNodes = getStartNodeNames();
        Set<String> leafNodes = getLeafNodeNames();

        if(startNodes.size() != 1){
            throw new Exception("DAG should have only one start node");
        }

        if(leafNodes.size() != 1){
            throw new Exception("DAG should have only one end node");
        }

        Set<String> zeroInDegree = startNodes;
        Set<String> executing = new HashSet<>();

        Map<String, Object> inputMap = new HashMap<>();

        for (String  s : zeroInDegree){
            nodes.get(s).updateInputDataMap("start", "0");
        }

        while(zeroInDegree.size() > 0){
            Set<String> readyToExecute = new HashSet<>(zeroInDegree);
            readyToExecute.removeAll(executing);
            executing.addAll(readyToExecute);

            ArrayList<NodeOutput> outputs= execute(readyToExecute);

            for(NodeOutput output: outputs ) {
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

        if(topoSortedList.size() != nodes.size()){
            throw new Exception("Not a valid DAG");
        }

        executorService.shutdown();
        return topoSortedList;

    }

    ArrayList<NodeOutput>  execute(Set<String> readyToExecute ){
        ArrayList<NodeOutput> out = new ArrayList<>();
        for(String name : readyToExecute){
            futures.add(executorCompletionService.submit(nodes.get(name)));
        }

        try {
            NodeOutput result = executorCompletionService.take().get();
            System.out.println("Result: " +result.getNodeName() + " "+ (String)result.getData());
            out.add(result);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return out;

    }

}
