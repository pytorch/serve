package org.pytorch.serve.checkpoint;

public class CheckpointReadException extends Exception {

    /** */
    private static final long serialVersionUID = 1L;

    public CheckpointReadException(String msg) {
        super(msg);
    }
}
