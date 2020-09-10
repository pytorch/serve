package org.pytorch.serve.openapi;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class Path {

    private Operation get;
    private Operation put;
    private Operation post;
    private Operation head;
    private Operation delete;
    private Operation patch;
    private Operation options;
    private List<Parameter> parameters;

}
