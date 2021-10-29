package org.pytorch.serve.ensemble;

public class NodeOutput {
    private String nodeName;
    private Object data;

    public NodeOutput(String nodeName, Object data) {
        this.nodeName = nodeName;
        this.data = data;
    }

    public String getNodeName() {
        return nodeName;
    }

    public void setNodeName(String nodeName) {
        this.nodeName = nodeName;
    }

    public Object getData() {
        return data;
    }

    public void setData(Object data) {
        this.data = data;
    }
}
