package org.pytorch.serve.openapi;

import java.util.LinkedHashMap;
import java.util.Map;

public class OpenApi {

    private String openapi = "3.0.1";
    private Info info;
    private Map<String, Path> paths;

    public OpenApi() {}

    public String getOpenapi() {
        return openapi;
    }

    public void setOpenapi(String openapi) {
        this.openapi = openapi;
    }

    public Info getInfo() {
        return info;
    }

    public void setInfo(Info info) {
        this.info = info;
    }

    public Map<String, Path> getPaths() {
        return paths;
    }

    public void setPaths(Map<String, Path> paths) {
        this.paths = paths;
    }

    public void addPath(String url, Path path) {
        if (paths == null) {
            paths = new LinkedHashMap<>();
        }
        paths.put(url, path);
    }
}
