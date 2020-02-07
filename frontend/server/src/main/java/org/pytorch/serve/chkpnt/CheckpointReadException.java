package org.pytorch.serve.chkpnt;

public class CheckpointReadException extends Exception {

    /** */
    private static final long serialVersionUID = 1L;

    public CheckpointReadException(String msg) {
        super(msg);
    }
}
