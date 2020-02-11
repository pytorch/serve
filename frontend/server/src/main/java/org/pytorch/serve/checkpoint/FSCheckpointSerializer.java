package org.pytorch.serve.checkpoint;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.http.ConflictStatusException;
import org.pytorch.serve.util.ConfigManager;

public class FSCheckpointSerializer implements CheckpointSerializer {

    private ConfigManager configManager = ConfigManager.getInstance();
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public void saveCheckpoint(Checkpoint checkpoint, Map<String, String> versionMarPath)
            throws IOException, ConflictStatusException {
        String chkpntJson = GSON.toJson(checkpoint, Checkpoint.class);

        String checkpointPath = configManager.getCheckpointStore() + "/" + checkpoint.getName();
        File checkPointModelStore = new File(checkpointPath + "/model_store");
        if (checkPointModelStore.exists()) {
            throw new ConflictStatusException(
                    "Checkpoint " + checkpoint.getName() + " already exists.");
        }
        checkPointModelStore.mkdirs();

        for (Map.Entry<String, String> marPath : versionMarPath.entrySet()) {
            String destMarFile = checkPointModelStore + "/" + marPath.getKey() + ".mar";
            FileOutputStream fos = new FileOutputStream(destMarFile);
            ZipOutputStream zos = new ZipOutputStream(fos);
            String modelFilesPath = marPath.getValue();
            for (String filename : new File(modelFilesPath).list()) {
                addDirToZipArchive(zos, new File(modelFilesPath + "/" + filename), null);
            }
            zos.flush();
            fos.flush();
            zos.close();
            fos.close();
        }
        try (FileWriter file =
                new FileWriter(checkpointPath + "/" + checkpoint.getName() + ".json")) {
            file.write(chkpntJson);
            file.flush();
        }
    }

    public Checkpoint getCheckpoint(String checkpointName) throws IOException {
        JsonParser jsonParser = new JsonParser();

        JsonObject checkpointJson = null;
        try (FileReader reader =
                new FileReader(
                        configManager.getCheckpointStore()
                                + "/"
                                + checkpointName
                                + "/"
                                + checkpointName
                                + ".json")) {
            checkpointJson = jsonParser.parse(reader).getAsJsonObject();
            Checkpoint checkpoint = GSON.fromJson(checkpointJson, Checkpoint.class);
            return checkpoint;
        }
    }

    public List<Checkpoint> getAllCheckpoints() throws IOException {

        ArrayList<Checkpoint> resp = new ArrayList<Checkpoint>();

        for (String checkPointName : new File(configManager.getCheckpointStore()).list()) {
            if (!(new File(configManager.getCheckpointStore() + "/" + checkPointName).isFile())) {
                resp.add(getCheckpoint(checkPointName));
            }
        }

        return resp;
    }

    public void removeCheckpoint(String checkpointName) throws IOException {
        String checkPointPath = configManager.getCheckpointStore() + "/" + checkpointName;
        FileUtils.deleteDirectory(new File(checkPointPath));
    }

    private void addDirToZipArchive(
            ZipOutputStream zos, File fileToZip, String parrentDirectoryName) throws IOException {
        if (fileToZip == null || !fileToZip.exists()) {
            return;
        }

        String zipEntryName = fileToZip.getName();
        if (parrentDirectoryName != null && !parrentDirectoryName.isEmpty()) {
            zipEntryName = parrentDirectoryName + "/" + fileToZip.getName();
        }

        if (fileToZip.isDirectory()) {
            if (!"__pycache__".equals(fileToZip.getName())) {
                for (File file : fileToZip.listFiles()) {
                    addDirToZipArchive(zos, file, zipEntryName);
                }
            }
        } else {
            byte[] buffer = new byte[1024];
            FileInputStream fis = new FileInputStream(fileToZip);
            zos.putNextEntry(new ZipEntry(zipEntryName));
            int length;
            while ((length = fis.read(buffer)) > 0) {
                zos.write(buffer, 0, length);
            }
            zos.closeEntry();
            fis.close();
        }
    }
}
