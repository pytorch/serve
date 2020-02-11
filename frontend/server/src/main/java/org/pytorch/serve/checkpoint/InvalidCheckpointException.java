package org.pytorch.serve.checkpoint;

public class InvalidCheckpointException extends Exception {

    private static final long serialVersionUID = 1L;

    public InvalidCheckpointException(String msg) {
        super(msg);
    }
}
