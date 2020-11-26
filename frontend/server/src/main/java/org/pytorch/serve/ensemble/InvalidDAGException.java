package org.pytorch.serve.ensemble;

public class InvalidDAGException extends Exception {
    public InvalidDAGException(String msg) {
        super(msg);
    }
}
