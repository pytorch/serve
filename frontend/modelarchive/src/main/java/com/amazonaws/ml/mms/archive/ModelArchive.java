/*
 * Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.amazonaws.ml.mms.archive;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Enumeration;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelArchive {

    private static final Logger logger = LoggerFactory.getLogger(ModelArchive.class);

    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private static final Pattern URL_PATTERN =
            Pattern.compile("http(s)?://.*", Pattern.CASE_INSENSITIVE);

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

    public static ModelArchive downloadModel(String modelStore, String url)
            throws ModelException, IOException {
        if (URL_PATTERN.matcher(url).matches()) {
            File modelDir = download(url);
            return load(url, modelDir, true);
        }

        if (url.contains("..")) {
            throw new ModelNotFoundException("Relative path is not allowed in url: " + url);
        }

        if (modelStore == null) {
            throw new ModelNotFoundException("Model store has not been configured.");
        }

        File modelLocation = new File(modelStore, url);
        if (!modelLocation.exists()) {
            throw new ModelNotFoundException("Model not found in model store: " + url);
        }
        if (modelLocation.isFile()) {
            try (InputStream is = new FileInputStream(modelLocation)) {
                File unzipDir = unzip(is, null);
                return load(url, unzipDir, true);
            }
        }
        return load(url, modelLocation, false);
    }

    public static void migrate(File legacyModelFile, File destination)
            throws InvalidModelException, IOException {
        boolean failed = true;
        try (ZipFile zip = new ZipFile(legacyModelFile);
                ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(destination))) {

            ZipEntry manifestEntry = zip.getEntry(MANIFEST_FILE);
            if (manifestEntry == null) {
                throw new InvalidModelException("Missing manifest file in model archive.");
            }

            InputStream is = zip.getInputStream(manifestEntry);
            Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8);

            JsonParser parser = new JsonParser();
            JsonObject json = (JsonObject) parser.parse(reader);
            JsonPrimitive version = json.getAsJsonPrimitive("specificationVersion");
            Manifest manifest;
            if (version != null && "1.0".equals(version.getAsString())) {
                throw new InvalidModelException("model archive is already in 1.0 version.");
            }

            LegacyManifest legacyManifest = GSON.fromJson(json, LegacyManifest.class);
            manifest = legacyManifest.migrate();

            zos.putNextEntry(new ZipEntry("MAR-INF/"));
            zos.putNextEntry(new ZipEntry("MAR-INF/" + MANIFEST_FILE));
            zos.write(GSON.toJson(manifest).getBytes(StandardCharsets.UTF_8));

            Enumeration<? extends ZipEntry> en = zip.entries();
            while (en.hasMoreElements()) {
                ZipEntry entry = en.nextElement();
                String name = entry.getName();
                if (MANIFEST_FILE.equalsIgnoreCase(name) || name.startsWith(".")) {
                    continue;
                }
                zos.putNextEntry(new ZipEntry(name));
                if (!entry.isDirectory()) {
                    IOUtils.copy(zip.getInputStream(entry), zos);
                }
            }
            failed = false;
        } finally {
            if (failed) {
                FileUtils.deleteQuietly(destination);
            }
        }
    }

    private static File download(String path) throws ModelException, IOException {
        HttpURLConnection conn;
        try {
            URL url = new URL(path);
            conn = (HttpURLConnection) url.openConnection();
            if (conn.getResponseCode() != HttpURLConnection.HTTP_OK) {
                throw new DownloadModelException(
                        "Failed to download model from: "
                                + path
                                + ", code: "
                                + conn.getResponseCode());
            }
        } catch (MalformedURLException | RuntimeException e) {
            // URLConnection may throw raw RuntimeException if port is out of range.
            throw new ModelNotFoundException("Invalid model url: " + path, e);
        } catch (IOException e) {
            throw new DownloadModelException("Failed to download model from: " + path, e);
        }

        try {
            String eTag = conn.getHeaderField("ETag");
            File tmpDir = new File(System.getProperty("java.io.tmpdir"));
            File modelDir = new File(tmpDir, "models");
            FileUtils.forceMkdir(modelDir);
            if (eTag != null) {
                if (eTag.startsWith("\"") && eTag.endsWith("\"") && eTag.length() > 2) {
                    eTag = eTag.substring(1, eTag.length() - 1);
                }
                File dir = new File(modelDir, eTag);
                if (dir.exists()) {
                    logger.info("model folder already exists: {}", eTag);
                    return dir;
                }
            }

            return unzip(conn.getInputStream(), eTag);
        } catch (SocketTimeoutException e) {
            throw new DownloadModelException("Download model timeout: " + path, e);
        }
    }

    private static ModelArchive load(String url, File dir, boolean extracted)
            throws InvalidModelException, IOException {
        boolean failed = true;
        try {
            File manifestFile = new File(dir, "MAR-INF/" + MANIFEST_FILE);
            Manifest manifest;
            if (manifestFile.exists()) {
                // Must be MMS 1.0 or later
                manifest = readFile(manifestFile, Manifest.class);
            } else {
                manifestFile = new File(dir, MANIFEST_FILE);
                boolean nested = false;
                if (!manifestFile.exists()) {
                    // Found MANIFEST.json in top level;
                    manifestFile = findFile(dir, MANIFEST_FILE, true); // for 0.1 model archive
                    nested = true;
                }

                if (manifestFile == null) {
                    // Must be 1.0
                    manifest = new Manifest();
                } else {
                    // 0.1 model may have extra parent directory
                    LegacyManifest legacyManifest = readFile(manifestFile, LegacyManifest.class);
                    manifest = legacyManifest.migrate();
                    File modelDir = manifestFile.getParentFile();
                    if (extracted && nested) {
                        // Move all file to top level, so we can clean up properly.
                        moveToTopLevel(modelDir, dir);
                    }
                }
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
        try (Reader r = new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8)) {
            return GSON.fromJson(r, type);
        } catch (JsonParseException e) {
            throw new InvalidModelException("Failed to parse signature.json.", e);
        }
    }

    private static File findFile(File dir, String fileName, boolean recursive) {
        File[] list = dir.listFiles();
        if (list == null) {
            return null;
        }
        for (File file : list) {
            if (recursive && file.isDirectory()) {
                File f = findFile(file, fileName, false);
                if (f != null) {
                    return f;
                }
            } else if (file.getName().equalsIgnoreCase(fileName)) {
                return file;
            }
        }
        return null;
    }

    private static void moveToTopLevel(File from, File to) throws IOException {
        File[] list = from.listFiles();
        if (list != null) {
            for (File file : list) {
                if (file.isDirectory()) {
                    FileUtils.moveDirectoryToDirectory(file, to, false);
                } else {
                    FileUtils.moveFileToDirectory(file, to, false);
                }
            }
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
            eTag = Hex.toHexString(md.digest());
        }
        File dir = new File(modelDir, eTag);
        if (dir.exists()) {
            FileUtils.deleteDirectory(tmp);
            logger.info("model folder already exists: {}", eTag);
            return dir;
        }

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

            if (model.getHandler() == null) {
                throw new InvalidModelException("Model handler is not defined.");
            }

            if (manifest.getRuntime() == null) {
                throw new InvalidModelException("Runtime is not defined or invalid.");
            }

            if (manifest.getEngine() != null && manifest.getEngine().getEngineName() == null) {
                throw new InvalidModelException("engineName is required in <engine>.");
            }
        } catch (InvalidModelException e) {
            clean();
            throw e;
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

    public void clean() {
        if (url != null && extracted) {
            FileUtils.deleteQuietly(modelDir);
        }
    }
}
