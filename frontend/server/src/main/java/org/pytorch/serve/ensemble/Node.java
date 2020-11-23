package org.pytorch.serve.ensemble;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;

public class Node<T> implements Callable {
    private String name;
    private String parentName;
    private Map<String, Object> inputDataMap;

    public WorkflowModel getWorkflowModel() {
        return workflowModel;
    }

    public void setWorkflowModel(WorkflowModel workflowModel) {
        this.workflowModel = workflowModel;
    }

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

    public Map<String, Object> getInputDataMap() {
        return inputDataMap;
    }

    public void updateInputDataMap(String s, Object inputData) {
        this.inputDataMap.put(s, inputData);
    }

    @Override
    public Object call() throws Exception {
        Random rand = new Random();
        // System.out.println("result  is "+(String)this.getInputDataMap() + rand.nextInt()+ " for
        // me "+ this.getName());
        ArrayList<String> a = new ArrayList<>();
        for (Object s : this.getInputDataMap().values()) {
            a.add((String) s);
        }

        a.add(rand.nextInt() + "");
        System.out.println(String.join("-", a));
        return new NodeOutput(this.getName(), String.join("-", a));
    }
}
