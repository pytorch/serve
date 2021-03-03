package org.pytorch.serve.http.messages;

public class KFV1ModelReadyResponse {

    private String name;
    private boolean ready;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public boolean getReady() {
        return ready;
    }

    public void setReady(boolean ready) {
        this.ready = ready;
    }
}
