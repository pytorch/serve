package org.pytorch.serve.openapi;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Info {

    private String title;
    private String description;
    private String termsOfService;
    private String version;

    public Info() {}
}
