package org.pytorch.serve.openapi;

public class Encoding {

    private String contentType;
    private String style;
    private boolean explode;
    private boolean allowReserved;

    public Encoding() {}

    public Encoding(String contentType) {
        this.contentType = contentType;
    }

    public String getContentType() {
        return contentType;
    }

    public void setContentType(String contentType) {
        this.contentType = contentType;
    }

    public boolean isAllowReserved() {
        return allowReserved;
    }

    public void setAllowReserved(boolean allowReserved) {
        this.allowReserved = allowReserved;
    }

    public String getStyle() {
        return style;
    }

    public void setStyle(String style) {
        this.style = style;
    }

    public boolean isExplode() {
        return explode;
    }

    public void setExplode(boolean explode) {
        this.explode = explode;
    }
}
