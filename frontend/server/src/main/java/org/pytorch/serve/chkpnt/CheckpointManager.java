package org.pytorch.serve.chkpnt;

import io.netty.handler.codec.http.HttpResponseStatus;
import java.util.List;

public class CheckpointManager {

    public static CheckpointManager getInstance() {
        return null;
    }

    public HttpResponseStatus saveCheckpoint(String chkpntName) {

        return null;
    }

    public List<String> getCheckpoints(String chkpntName) {

        return null;
    }

    public HttpResponseStatus restartwithCheckpoint(String chkpntName) {
        return null;
    }

    public HttpResponseStatus removeCheckpoint(String chkpntName) {
        return null;
    }
}
