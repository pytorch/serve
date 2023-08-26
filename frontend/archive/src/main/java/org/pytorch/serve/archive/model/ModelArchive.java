package org.pytorch.serve.archive.model;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.utils.ArchiveUtils;
import org.pytorch.serve.archive.utils.InvalidArchiveURLException;
import org.pytorch.serve.archive.utils.ZipUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelArchive {

    private static final Logger logger = LoggerFactory.getLogger(ModelArchive.class);

    private static final String MANIFEST_FILE = "MANIFEST.json";

    private Manifest manifest;
    private String url;
    private File modelDir;
    private boolean extracted;
    private ModelConfig modelConfig;

    public ModelArchive(Manifest manifest, String url, File modelDir, boolean extracted) {
        this.manifest = manifest;
        this.url = url;
        this.modelDir = modelDir;
        this.extracted = extracted;
        this.modelConfig = null;
    }

    public static ModelArchive downloadModel(
            List<String> allowedUrls, String modelStore, String url)
            throws ModelException, FileAlreadyExistsException, IOException,
                    DownloadArchiveException {
        return downloadModel(allowedUrls, modelStore, url, false);
    }

    public static ModelArchive downloadModel(
            List<String> allowedUrls, String modelStore, String url, boolean s3SseKmsEnabled)
            throws ModelException, FileAlreadyExistsException, IOException,
                    DownloadArchiveException {
        if (modelStore == null) {
            throw new ModelNotFoundException("Model store has not been configured.");
        }

        if (url == null || url.isEmpty()) {
            throw new ModelNotFoundException("empty url");
        }

        String marFileName = ArchiveUtils.getFilenameFromUrl(url);
        File modelLocation = new File(modelStore, marFileName);
        try {
            ArchiveUtils.downloadArchive(
                    allowedUrls, modelLocation, marFileName, url, s3SseKmsEnabled);
        } catch (InvalidArchiveURLException e) {
            throw new ModelNotFoundException(e.getMessage()); // NOPMD
        }

        if (url.contains("..")) {
            throw new ModelNotFoundException("Relative path is not allowed in url: " + url);
        }

        if (modelLocation.isFile()) {
            try (InputStream is = Files.newInputStream(modelLocation.toPath())) {
                File unzipDir;
                if (modelLocation.getName().endsWith(".mar")) {
                    unzipDir = ZipUtils.unzip(is, null, "models", true);
                } else {
                    unzipDir = ZipUtils.unzip(is, null, "models", false);
                }
                return load(url, unzipDir, true);
            }
        }

        File tempDir = ZipUtils.createTempDir(null, "models");
        logger.info("createTempDir {}", tempDir.getAbsolutePath());
        File directory = new File(url);
        if (directory.isDirectory()) {
            // handle the case that the input url is a directory.
            // the input of url is "/xxx/model_store/modelXXX" or
            // "xxxx/yyyyy/modelXXX".
            File[] fileList = directory.listFiles();
            if (fileList.length == 1 && fileList[0].isDirectory()) {
                // handle the case that a model tgz file
                // has root dir after decompression on SageMaker
                File targetLink = ZipUtils.createSymbolicDir(fileList[0], tempDir);
                logger.info("createSymbolicDir {}", targetLink.getAbsolutePath());
                return load(url, targetLink, false);
            }
            File targetLink = ZipUtils.createSymbolicDir(directory, tempDir);
            logger.info("createSymbolicDir {}", targetLink.getAbsolutePath());
            return load(url, targetLink, false);
        } else if (modelLocation.exists()) {
            // handle the case that "/xxx/model_store/modelXXX" is directory.
            // the input of url is modelXXX when torchserve is started
            // with snapshot or with parameter --models modelXXX
            File[] fileList = modelLocation.listFiles();
            if (fileList.length == 1 && fileList[0].isDirectory()) {
                // handle the case that a model tgz file
                // has root dir after decompression on SageMaker
                File targetLink = ZipUtils.createSymbolicDir(fileList[0], tempDir);
                logger.info("createSymbolicDir {}", targetLink.getAbsolutePath());
                return load(url, targetLink, false);
            }
            File targetLink = ZipUtils.createSymbolicDir(modelLocation, tempDir);
            logger.info("createSymbolicDir {}", targetLink.getAbsolutePath());
            return load(url, targetLink, false);
        }

        throw new ModelNotFoundException("Model not found at: " + url);
    }

    private static ModelArchive load(String url, File dir, boolean extracted)
            throws InvalidModelException, IOException {
        boolean failed = true;
        try {
            File manifestFile = new File(dir, "MAR-INF/" + MANIFEST_FILE);
            Manifest manifest;
            if (manifestFile.exists()) {
                manifest = ArchiveUtils.readFile(manifestFile, Manifest.class);
            } else {
                manifest = new Manifest();
            }

            failed = false;
            return new ModelArchive(manifest, url, dir, extracted);
        } finally {
            if (failed) {
                if (Files.isSymbolicLink(dir.toPath())) {
                    FileUtils.deleteQuietly(dir.getParentFile());
                } else {
                    FileUtils.deleteQuietly(dir);
                }
            }
        }
    }

    public void validate() throws InvalidModelException {
        Manifest.Model model = manifest.getModel();
        try {
            if (model == null) {
                throw new InvalidModelException("Missing Model entry in manifest file.");
            }

            if (model.getModelName() == null) {
                throw new InvalidModelException("Model name is not defined.");
            }

            if (model.getModelVersion() == null) {
                throw new InvalidModelException("Model version is not defined.");
            }

            if (manifest.getRuntime() == null) {
                throw new InvalidModelException("Runtime is not defined or invalid.");
            }

            if (manifest.getArchiverVersion() == null) {
                logger.warn(
                        "Model archive version is not defined. Please upgrade to torch-model-archiver 0.2.0 or higher");
            }

            if (manifest.getCreatedOn() == null) {
                logger.warn(
                        "Model archive createdOn is not defined. Please upgrade to torch-model-archiver 0.2.0 or higher");
            }
        } catch (InvalidModelException e) {
            clean();
            throw e;
        }
    }

    public static void removeModel(String modelStore, String marURL) {
        if (ArchiveUtils.isValidURL(marURL)) {
            String marFileName = ArchiveUtils.getFilenameFromUrl(marURL);
            File modelLocation = new File(modelStore, marFileName);
            FileUtils.deleteQuietly(modelLocation);
        }
    }

    public String getHandler() {
        return manifest.getModel().getHandler();
    }

    public Manifest getManifest() {
        return manifest;
    }

    public String getUrl() {
        return url;
    }

    public File getModelDir() {
        return modelDir;
    }

    public String getModelName() {
        return manifest.getModel().getModelName();
    }

    public String getModelVersion() {
        return manifest.getModel().getModelVersion();
    }

    public void clean() {
        if (url != null) {
            if (Files.isSymbolicLink(modelDir.toPath())) {
                FileUtils.deleteQuietly(modelDir.getParentFile());
            } else {
                FileUtils.deleteQuietly(modelDir);
            }
        }
    }

    public ModelConfig getModelConfig() {
        if (this.modelConfig == null && manifest.getModel().getConfigFile() != null) {
            try {
                File configFile =
                        new File(modelDir.getAbsolutePath(), manifest.getModel().getConfigFile());
                Map<String, Object> modelConfigMap = ArchiveUtils.readYamlFile(configFile);
                this.modelConfig = ModelConfig.build(modelConfigMap);
            } catch (InvalidModelException | IOException e) {
                logger.error(
                        "Failed to parse model config file {}",
                        manifest.getModel().getConfigFile(),
                        e);
            }
        }
        return this.modelConfig;
    }
}
