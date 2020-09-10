package org.pytorch.serve.openapi;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Encoding {

    private String contentType;
    private String style;
    private boolean explode;
    private boolean allowReserved;

    public Encoding() {}

    public Encoding(String contentType) {
        this.contentType = contentType;
    }
}
