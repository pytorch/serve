package org.pytorch.serve.snapshot;

public class InvalidSnapshotException extends Exception {

    private static final long serialVersionUID = 1L;

    public InvalidSnapshotException(String msg) {
        super(msg);
    }
}
