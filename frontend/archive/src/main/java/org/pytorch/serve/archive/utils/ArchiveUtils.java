package org.pytorch.serve.archive.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.util.List;
import java.util.regex.Pattern;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.archive.DownloadArchiveException;
import org.pytorch.serve.archive.model.InvalidModelException;
import org.pytorch.serve.archive.s3.HttpUtils;

public final class ArchiveUtils {

    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private static final Pattern VALID_URL_PATTERN =
            Pattern.compile("file?://.*|http(s)?://.*", Pattern.CASE_INSENSITIVE);

    private ArchiveUtils() {}

    public static <T> T readFile(File file, Class<T> type)
            throws InvalidModelException, IOException {
        try (Reader r =
                new InputStreamReader(
                        Files.newInputStream(file.toPath()), StandardCharsets.UTF_8)) {
            return GSON.fromJson(r, type);
        } catch (JsonParseException e) {
            throw new InvalidModelException("Failed to parse signature.json.", e);
        }
    }

    public static boolean validateURL(List<String> allowedUrls, String url)
            throws InvalidArchiveURLException {
        boolean patternMatch = false;
        for (String temp : allowedUrls) {
            if (Pattern.compile(temp, Pattern.CASE_INSENSITIVE).matcher(url).matches()) {
                patternMatch = true;
                return patternMatch;
            }
        }
        if (isValidURL(url)) {
            // case when url is valid url but does not match valid hosts
            throw new InvalidArchiveURLException(
                    "Given URL " + url + " does not match any allowed URL(s)");
        }
        return patternMatch;
    }

    public static boolean isValidURL(String url) {
        return VALID_URL_PATTERN.matcher(url).matches();
    }

    public static boolean downloadArchive(
            List<String> allowedUrls,
            File location,
            String archiveName,
            String url,
            boolean s3SseKmsEnabled)
            throws FileAlreadyExistsException, FileNotFoundException, DownloadArchiveException,
                    InvalidArchiveURLException {
        if (validateURL(allowedUrls, url)) {
            if (location.exists()) {
                throw new FileAlreadyExistsException(archiveName);
            }
            try {
                HttpUtils.copyURLToFile(new URL(url), location, s3SseKmsEnabled);
            } catch (IOException e) {
                FileUtils.deleteQuietly(location);
                throw new DownloadArchiveException("Failed to download archive from: " + url, e);
            }
        }

        return true;
    }
}
