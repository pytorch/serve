package org.pytorch.serve.archive.workflow;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.utils.ArchiveUtils;
import org.pytorch.serve.archive.utils.InvalidArchiveURLException;
import org.pytorch.serve.archive.utils.ZipUtils;

public class WorkflowArchive {

    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private static final String MANIFEST_FILE = "MANIFEST.json";

    private Manifest manifest;
    private String url;
    private File workflowDir;
    private boolean extracted;

    public WorkflowArchive(Manifest manifest, String url, File workflowDir, boolean extracted) {
        this.manifest = manifest;
        this.url = url;
        this.workflowDir = workflowDir;
        this.extracted = extracted;
    }

    public static WorkflowArchive downloadWorkflow(
            List<String> allowedUrls, String workflowStore, String url)
            throws WorkflowException, FileAlreadyExistsException, IOException,
                    DownloadArchiveException {
        return downloadWorkflow(allowedUrls, workflowStore, url, false);
    }

    public static WorkflowArchive downloadWorkflow(
            List<String> allowedUrls, String workflowStore, String url, boolean s3SseKmsEnabled)
            throws WorkflowException, FileAlreadyExistsException, IOException,
                    DownloadArchiveException {

        if (workflowStore == null) {
            throw new WorkflowNotFoundException("Workflow store has not been configured.");
        }

        String warFileName = FilenameUtils.getName(url);
        File workflowLocation = new File(workflowStore, warFileName);

        try {
            ArchiveUtils.downloadArchive(
                    allowedUrls, workflowLocation, warFileName, url, s3SseKmsEnabled);
        } catch (InvalidArchiveURLException e) {
            throw new WorkflowNotFoundException(e.getMessage()); // NOPMD
        }

        if (url.contains("..")) {
            throw new WorkflowNotFoundException("Relative path is not allowed in url: " + url);
        }

        if (!workflowLocation.exists()) {
            throw new WorkflowNotFoundException("Workflow not found in workflow store: " + url);
        }

        if (workflowLocation.isFile()) {
            try (InputStream is = Files.newInputStream(workflowLocation.toPath())) {
                File unzipDir = ZipUtils.unzip(is, null, "workflows");
                return load(url, unzipDir, true);
            }
        }

        throw new WorkflowNotFoundException("Workflow not found at: " + url);
    }

    private static WorkflowArchive load(String url, File dir, boolean extracted)
            throws InvalidWorkflowException, IOException {
        boolean failed = true;
        try {
            File manifestFile = new File(dir, "WAR-INF/" + MANIFEST_FILE);
            Manifest manifest = null;
            if (manifestFile.exists()) {
                manifest = readFile(manifestFile, Manifest.class);
            } else {
                manifest = new Manifest();
            }

            failed = false;
            return new WorkflowArchive(manifest, url, dir, extracted);
        } finally {
            if (extracted && failed) {
                FileUtils.deleteQuietly(dir);
            }
        }
    }

    private static <T> T readFile(File file, Class<T> type)
            throws InvalidWorkflowException, IOException {
        try (Reader r =
                new InputStreamReader(
                        Files.newInputStream(file.toPath()), StandardCharsets.UTF_8)) {
            return GSON.fromJson(r, type);
        } catch (JsonParseException e) {
            throw new InvalidWorkflowException("Failed to parse signature.json.", e);
        }
    }

    public void validate() throws InvalidWorkflowException {
        Manifest.Workflow workflow = manifest.getWorkflow();
        try {
            if (workflow == null) {
                throw new InvalidWorkflowException("Missing Workflow entry in manifest file.");
            }

            if (workflow.getWorkflowName() == null) {
                throw new InvalidWorkflowException("Workflow name is not defined.");
            }

            if (manifest.getArchiverVersion() == null) {
                throw new InvalidWorkflowException("Workflow archive version is not defined.");
            }

            if (manifest.getCreatedOn() == null) {
                throw new InvalidWorkflowException("Workflow archive createdOn is not defined.");
            }
        } catch (InvalidWorkflowException e) {
            clean();
            throw e;
        }
    }

    public static void removeWorkflow(String workflowStore, String warURL) {
        if (ArchiveUtils.isValidURL(warURL)) {
            String warFileName = FilenameUtils.getName(warURL);
            File workflowLocation = new File(workflowStore, warFileName);
            FileUtils.deleteQuietly(workflowLocation);
        }
    }

    public String getHandler() {
        return manifest.getWorkflow().getHandler();
    }

    public Manifest getManifest() {
        return manifest;
    }

    public String getUrl() {
        return url;
    }

    public File getWorkflowDir() {
        return workflowDir;
    }

    public String getWorkflowName() {
        return manifest.getWorkflow().getWorkflowName();
    }

    public void clean() {
        if (url != null && extracted) {
            FileUtils.deleteQuietly(workflowDir);
        }
    }
}
