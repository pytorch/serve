package org.pytorch.serve.chkpnt;

public class InvalidCheckPointException extends Exception {

    private static final long serialVersionUID = 1L;

    public InvalidCheckPointException(String msg) {
        super(msg);
    }
}
