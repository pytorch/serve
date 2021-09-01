package org.pytorch.serve.util.messages;

import java.util.List;

public class ModelWorkerResponse {

    private List<Integer> codes;
    private List<String> messages;
    private List<Predictions> predictions;

    public ModelWorkerResponse() {}

    public boolean isAll200Code() {
        for (Integer code : codes) {
            if(code != 200) {
                return false;
            }
        }
        return true;
    }

    public void setCodes(List<Integer> codes) {
        this.codes = codes;
    }

    public void appendCode(Integer code) {
        this.codes.add(code);
    }

    public List<Integer> getCodes() {
        return this.codes;
    }

    public List<String> getMessages() {
        return this.messages;
    }

    public void appendMessage(String message) {
        this.messages.add(message);
    }

    public void setMessages(List<String> messages) {
        this.messages = messages;
    }

    public List<Predictions> getPredictions() {
        return predictions;
    }

    public void setPredictions(List<Predictions> predictions) {
        this.predictions = predictions;
    }

    public void appendPredictions(Predictions prediction) {
        this.predictions.add(prediction);
    }
}
