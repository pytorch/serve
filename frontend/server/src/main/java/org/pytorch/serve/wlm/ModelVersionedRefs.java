package org.pytorch.serve.wlm;

import org.pytorch.serve.archive.Manifest;
import org.pytorch.serve.archive.ModelArchive;
import org.pytorch.serve.archive.ModelException;
import org.pytorch.serve.archive.ModelNotFoundException;
import org.pytorch.serve.http.InvalidModelVersionException;
import org.pytorch.serve.util.ConfigManager;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ModelVersionedRefs {
    private static final Logger logger = LoggerFactory.getLogger(ModelVersionedRefs.class);

    private ConcurrentHashMap<Double, Model> modelsVersionMap;
    Double defaultVersion;

    public ModelVersionedRefs() {
        this.modelsVersionMap = new ConcurrentHashMap<>();
    }

    private void _validateVersionId(String v)
	throws InvalidModelVersionException, NumberFormatException {
	Double vd = Double.valueOf(v);
	if (vd <= Double.valueOf("0.0"))
	    throw new InvalidModelVersionException("Model Version is invalid: " + v);
    }

    private void _checkVersionCapacity() {
	// place holder only for now
    }

    /**
     * Adds a new version of the Model to the Map if it does not exist
     * Sets this version as the default version of the model which is
     * automatically served on the next request to this model.
     * If it already exists in the map, throws an exception with conflict
     * status
     *
     * @param model: Model object with all the parameters initialized
     *         as desired
     * @param versionId: String obj of version ID from the manifest
     * @return None
     */
    public void addVersionModel(Model model, String versionId)
	throws InvalidModelVersionException, InvalidModelVersionException {
	_validateVersionId(versionId);
	_checkVersionCapacity();
	if (this.modelsVersionMap.putIfAbsent(Double.valueOf(versionId), model) != null)
	    throw new InvalidModelVersionException(
						   "Model " + model.getModelName() + " is already registered.");
	this.setDefaultVersion(versionId);
    }

    /**
     * Returns a String object of the default version of this Model
     * @return      String obj of the current default Version
     */
    public String getDefaultVersion() {
	return this.defaultVersion.toString();
    }

    /**
     * Sets the default version of the model to the version in
     * arg
     * @param       A valid String obj with version to set default
     * @return      None
     */
    public void setDefaultVersion(String versionId)
	throws InvalidModelVersionException {
	_validateVersionId(versionId);
	if (this.modelsVersionMap.get(Double.valueOf(versionId)) == null)
	    throw new InvalidModelVersionException("Can't set default to: " + versionId);
	this.defaultVersion = Double.valueOf(versionId);
    }

    /**
     * Removes the specified version of the model from the Map
     * If it's the default version then throws an exception
     * The Client is responsible for setting a new default
     * prior to deleting the current default
     *
     * @param  A String specifying a valid non-default version Id
     * @return On Success - a String specifying the new default version Id
     *         On Failure - throws InvalidModelVersionException
     */
    public String removeVersionModel(String versionId)
	throws InvalidModelVersionException {
	if (this.defaultVersion.compareTo(Double.valueOf(versionId)) == 0) {
	    throw new InvalidModelVersionException("Can't remove default version: " + versionId);
	}
	return this.defaultVersion.toString();
    }

    /**
     * Returns the Model obj corresponding to the version provided
     *
     * @param  A String specifying a valid version Id
     * @return On Success - a Model Obj previously registered
     *         On Failure - null
     */
    public Model getVersionModel(String versionId)
	throws InvalidModelVersionException {
	_validateVersionId(versionId);
	return this.modelsVersionMap.get(Double.valueOf(versionId));
    }


    /**
     * Returns the default Model obj
     *
     * @param  None
     * @return On Success - a Model Obj corresponding to the default Model obj
     *         On Failure - null
     */
    public Model getDefaultModel()
	throws InvalidModelVersionException {
	return this.modelsVersionMap.get(this.defaultVersion);
    }

    // scope for a nice generator pattern impl here
    public Model forAllVersions() {
	return null;
    }

    public Set<Map.Entry<Double, Model>> getAllVersions() {
	return this.modelsVersionMap.entrySet();
    }
}
