package org.pytorch.serve.wlm;

public class ModelVersionName {

    private String modelName;
    private String version;

    public ModelVersionName(String modelName, String version) {
        this.modelName = modelName;
        this.version = version;
    }

    public String getModelName() {
        return modelName;
    }

    public String getVersion() {
        return version;
    }

    public String getVersionedModelName() {
        return getModelName() + "_" + getVersion();
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((modelName == null) ? 0 : modelName.hashCode());
        result = prime * result + ((version == null) ? 0 : version.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }

        if (!(obj instanceof ModelVersionName)) {
            return false;
        }

        ModelVersionName mvn = (ModelVersionName) obj;

        return (mvn.getModelName().equals(this.modelName))
                && (mvn.getVersion().equals(this.version));
    }
}
