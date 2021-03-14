package org.pytorch.serve.archive;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.UUID;
import java.util.regex.Pattern;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelArchive {

    private static final Logger logger = LoggerFactory.getLogger(ModelArchive.class);

    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private static final Pattern VALID_URL_PATTERN =
            Pattern.compile("file?://.*|http(s)?://.*", Pattern.CASE_INSENSITIVE);

    private static final String MANIFEST_FILE = "MANIFEST.json";

    private Manifest manifest;
    private String url;
    private File modelDir;
    private boolean extracted;

    public ModelArchive(Manifest manifest, String url, File modelDir, boolean extracted) {
        this.manifest = manifest;
        this.url = url;
        this.modelDir = modelDir;
        this.extracted = extracted;
    }

    public static ModelArchive downloadModel(
            List<String> allowedUrls, String modelStore, String url)
            throws ModelException, FileAlreadyExistsException, IOException {

        if (modelStore == null) {
            throw new ModelNotFoundException("Model store has not been configured.");
        }

        String marFileName = FilenameUtils.getName(url);
        File modelLocation = new File(modelStore, marFileName);
        if (checkAllowedUrl(allowedUrls, url)) {
            if (modelLocation.exists()) {
                throw new FileAlreadyExistsException(marFileName);
            }
            try {
                FileUtils.copyURLToFile(new URL(url), modelLocation);
            } catch (IOException e) {
                FileUtils.deleteQuietly(modelLocation);
                throw new DownloadModelException("Failed to download model from: " + url, e);
            }
        }

        if (url.contains("..")) {
            throw new ModelNotFoundException("Relative path is not allowed in url: " + url);
        }

        if (modelLocation.isFile()) {
            try (InputStream is = Files.newInputStream(modelLocation.toPath())) {
                File unzipDir = unzip(is, null);
                return load(url, unzipDir, true);
            }
        }

        if (new File(url).isDirectory()) {
            return load(url, new File(url), false);
        }

        throw new ModelNotFoundException("Model not found at: " + url);
    }

    public static boolean checkAllowedUrl(List<String> allowedUrls, String url)
            throws ModelNotFoundException {
        boolean patternMatch = false;
        for (String temp : allowedUrls) {
            if (Pattern.compile(temp, Pattern.CASE_INSENSITIVE).matcher(url).matches()) {
                patternMatch = true;
                return patternMatch;
            }
        }
        if (VALID_URL_PATTERN.matcher(url).matches()) {
            // case when url is valid url but does not match valid hosts
            throw new ModelNotFoundException(
                    "Given URL " + url + " does not match any allowed URL(s)");
        }
        return patternMatch;
    }

    private static ModelArchive load(String url, File dir, boolean extracted)
            throws InvalidModelException, IOException {
        boolean failed = true;
        try {
            File manifestFile = new File(dir, "MAR-INF/" + MANIFEST_FILE);
            Manifest manifest = null;
            if (manifestFile.exists()) {
                manifest = readFile(manifestFile, Manifest.class);
            } else {
                manifest = new Manifest();
            }

            failed = false;
            return new ModelArchive(manifest, url, dir, extracted);
        } finally {
            if (extracted && failed) {
                FileUtils.deleteQuietly(dir);
            }
        }
    }

    private static <T> T readFile(File file, Class<T> type)
            throws InvalidModelException, IOException {
        try (Reader r =
                new InputStreamReader(
                        Files.newInputStream(file.toPath()), StandardCharsets.UTF_8)) {
            return GSON.fromJson(r, type);
        } catch (JsonParseException e) {
            throw new InvalidModelException("Failed to parse signature.json.", e);
        }
    }

    public static File unzip(InputStream is, String eTag) throws IOException {
        File tmpDir = FileUtils.getTempDirectory();
        File modelDir = new File(tmpDir, "models");
        FileUtils.forceMkdir(modelDir);

        File tmp = File.createTempFile("model", ".download");
        FileUtils.forceDelete(tmp);
        FileUtils.forceMkdir(tmp);

        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA1");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
        ZipUtils.unzip(new DigestInputStream(is, md), tmp);
        if (eTag == null) {
            eTag = UUID.randomUUID().toString().replaceAll("-", "");
        }
        logger.info("eTag {}", eTag);
        File dir = new File(modelDir, eTag);

        FileUtils.moveDirectory(tmp, dir);

        return dir;
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
        if (VALID_URL_PATTERN.matcher(marURL).matches()) {
            String marFileName = FilenameUtils.getName(marURL);
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
        if (url != null && extracted) {
            FileUtils.deleteQuietly(modelDir);
        }
    }
}
