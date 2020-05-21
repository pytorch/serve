package org.pytorch.serve.wlm;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.pytorch.serve.archive.ModelVersionNotFoundException;
import org.pytorch.serve.http.ConflictStatusException;
import org.pytorch.serve.http.InvalidModelVersionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ModelVersionedRefs {
    private static final Logger logger = LoggerFactory.getLogger(ModelVersionedRefs.class);

    private ConcurrentHashMap<String, Model> modelsVersionMap;
    private String defaultVersion;

    public ModelVersionedRefs() {
        this.modelsVersionMap = new ConcurrentHashMap<>();
    }

    /**
     * Adds a new version of the Model to the Map if it does not exist Sets this version as the
     * default version of the model which is automatically served on the next request to this model.
     * If it already exists in the map, throws an exception with conflict status
     *
     * @param model: Model object with all the parameters initialized as desired
     * @param versionId: String version ID from the manifest
     * @return None
     */
    public void addVersionModel(Model model, String versionId)
            throws ModelVersionNotFoundException, ConflictStatusException {
        logger.debug("Adding new version {} for model {}", versionId, model.getModelName());

        if (versionId == null) {
            throw new InvalidModelVersionException("Model version not found. ");
        }

        if (this.modelsVersionMap.putIfAbsent(versionId, model) != null) {
            throw new ConflictStatusException(
                    "Model version "
                            + versionId
                            + " is already registered for model "
                            + model.getModelName());
        }

        if (this.defaultVersion == null) {
            this.setDefaultVersion(versionId);
        }
    }

    /**
     * Returns a String object of the default version of this Model
     *
     * @return String obj of the current default Version
     */
    public String getDefaultVersion() {
        return this.defaultVersion;
    }

    /**
     * Sets the default version of the model to the version in arg
     *
     * @param A valid String obj with version to set default
     * @return None
     */
    public void setDefaultVersion(String versionId) throws ModelVersionNotFoundException {
        Model model = this.modelsVersionMap.get(versionId);
        if (model == null) {
            throw new ModelVersionNotFoundException("Can't set default to: " + versionId);
        }

        logger.debug("Setting default version to {} for model {}", versionId, model.getModelName());
        this.defaultVersion = versionId;
    }

    /**
     * Removes the specified version of the model from the Map If it's the default version then
     * throws an exception The Client is responsible for setting a new default prior to deleting the
     * current default
     *
     * @param A String specifying a valid non-default version Id
     * @return On Success - Removed model for given version Id
     * @throws ModelVersionNotFoundException
     * @throws On Failure - throws InvalidModelVersionException and ModelNotFoundException
     */
    public Model removeVersionModel(String versionId)
            throws InvalidModelVersionException, ModelVersionNotFoundException {
        if (versionId == null) {
            versionId = this.getDefaultVersion();
        }

        if (this.defaultVersion.equals(versionId) && modelsVersionMap.size() > 1) {
            throw new InvalidModelVersionException(
                    String.format("Can't remove default version: %s", versionId));
        }

        Model model = this.modelsVersionMap.remove(versionId);
        if (model == null) {
            throw new ModelVersionNotFoundException(
                    String.format("Model version: %s not found", versionId));
        }

        logger.debug("Removed model: {} version: {}", model.getModelName(), versionId);

        return model;
    }

    /**
     * Returns the Model obj corresponding to the version provided
     *
     * @param A String specifying a valid version Id
     * @return On Success - a Model Obj previously registered On Failure - null
     */
    public Model getVersionModel(String versionId) {
        Model model = null;
        if (versionId != null) {
            model = this.modelsVersionMap.get(versionId);
        } else {
            model = this.getDefaultModel();
        }

        return model;
    }

    /**
     * Returns the default Model obj
     *
     * @param None
     * @return On Success - a Model Obj corresponding to the default Model obj On Failure - null
     */
    public Model getDefaultModel() {
        // TODO should not throw invalid here as it has been already validated??
        return this.modelsVersionMap.get(this.defaultVersion);
    }

    // scope for a nice generator pattern impl here
    // TODO what is this for?
    public Model forAllVersions() {
        return null;
    }

    public Set<Map.Entry<String, Model>> getAllVersions() {
        return this.modelsVersionMap.entrySet();
    }
}
