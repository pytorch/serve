package org.pytorch.serve.ensemble;

public class InvalidDAGException extends Exception {

    private static final long serialVersionUID = 1L;

    public InvalidDAGException(String msg) {
        super(msg);
    }
}
