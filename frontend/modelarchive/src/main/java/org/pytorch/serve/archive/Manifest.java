package org.pytorch.serve.archive;

import com.google.gson.annotations.SerializedName;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Manifest {

    private String createdOn;
    private String description;
    private String archiverVersion;
    private RuntimeType runtime;
    private Model model;

    public Manifest() {
        runtime = RuntimeType.PYTHON;
        model = new Model();
    }

    @Getter
    @Setter
    public static final class Model {

        private String modelName;
        private String version;
        private String description;
        private String modelVersion;
        private String handler;
        private String requirementsFile;

        public Model() {}
    }

    public enum RuntimeType {
        @SerializedName("python")
        PYTHON("python"),
        @SerializedName("python2")
        PYTHON2("python2"),
        @SerializedName("python3")
        PYTHON3("python3");

        String value;

        RuntimeType(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        public static RuntimeType fromValue(String value) {
            for (RuntimeType runtime : values()) {
                if (runtime.value.equals(value)) {
                    return runtime;
                }
            }
            throw new IllegalArgumentException("Invalid RuntimeType value: " + value);
        }
    }
}
