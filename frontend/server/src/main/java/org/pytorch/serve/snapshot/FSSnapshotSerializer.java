package org.pytorch.serve.snapshot;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Comparator;
import java.util.Date;
import java.util.Optional;
import java.util.Properties;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.servingsdk.snapshot.Snapshot;
import org.pytorch.serve.servingsdk.snapshot.SnapshotSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FSSnapshotSerializer implements SnapshotSerializer {

    private Logger logger = LoggerFactory.getLogger(FSSnapshotSerializer.class);
    private static final String MODEL_SNAPSHOT = "model_snapshot";
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    @Override
    public void saveSnapshot(Snapshot snapshot, final Properties prop) throws IOException {
        File snapshotPath = new File(getSnapshotDirectory());

        FileUtils.forceMkdir(snapshotPath);

        File snapshotFile = new File(snapshotPath, snapshot.getName());
        if (snapshotFile.exists()) {
            logger.error(
                    "Snapshot " + snapshot.getName() + " already exists. Not saving the sanpshot.");
            return;
        }

        String snapshotJson = GSON.toJson(snapshot, Snapshot.class);
        prop.put(MODEL_SNAPSHOT, snapshotJson);
        try (OutputStream os = Files.newOutputStream(snapshotFile.toPath())) {
            OutputStreamWriter osWriter = new OutputStreamWriter(os, StandardCharsets.UTF_8);
            prop.store(osWriter, "Saving snapshot");
            osWriter.flush();
            osWriter.close();
        }
    }

    @Override
    public Snapshot getSnapshot(String snapshotJson) throws IOException {
        return GSON.fromJson(snapshotJson, Snapshot.class);
    }

    public static String getSnapshotPath(String snapshotName) {
        return getSnapshotDirectory() + "/" + snapshotName;
    }

    public static String getSnapshotDirectory() {
        return System.getProperty("LOG_LOCATION") + "/config";
    }

    @Override
    public Properties getLastSnapshot() {
        String latestSnapshotPath = null;
        Path configPath = Paths.get(FSSnapshotSerializer.getSnapshotDirectory());

        if (Files.exists(configPath)) {
            try {
                Optional<Path> lastFilePath =
                        Files.list(configPath)
                                .filter(f -> !Files.isDirectory(f))
                                .max(
                                        Comparator.comparingLong(
                                                f -> getSnapshotTime(f.getFileName().toString())));
                if (lastFilePath.isPresent()) {
                    latestSnapshotPath = lastFilePath.get().toString();
                }
            } catch (IOException e) {
                e.printStackTrace(); // NOPMD
            }
        }

        return loadProperties(latestSnapshotPath);
    }

    private Properties loadProperties(String propPath) {
        if (propPath != null) {
            File propFile = new File(propPath);
            try (InputStream stream = Files.newInputStream(propFile.toPath())) {
                Properties prop = new Properties();
                prop.load(stream);
                prop.put("tsConfigFile", propPath);
                return prop;
            } catch (IOException e) {
                e.printStackTrace(); // NOPMD
            }
        }
        return null;
    }

    private static long getSnapshotTime(String filename) {
        String timestamp = filename.split("-")[0];
        Date d = null;
        try {
            d = new SimpleDateFormat("yyyyMMddHHmmssSSS").parse(timestamp);
        } catch (ParseException e) {
            e.printStackTrace(); // NOPMD
        }
        return d.getTime();
    }
}
