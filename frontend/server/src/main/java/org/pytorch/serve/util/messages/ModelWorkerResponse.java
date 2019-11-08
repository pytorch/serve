package org.pytorch.serve.util.messages;

import java.util.List;

public class ModelWorkerResponse {

    private int code;
    private String message;
    private List<Predictions> predictions;

    public ModelWorkerResponse() {}

    public int getCode() {
        return code;
    }

    public void setCode(int code) {
        this.code = code;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
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
