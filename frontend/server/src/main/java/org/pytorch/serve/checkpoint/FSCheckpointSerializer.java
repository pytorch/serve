package org.pytorch.serve.checkpoint;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import org.apache.commons.io.FileUtils;
import org.pytorch.serve.http.ConflictStatusException;
import org.pytorch.serve.util.ConfigManager;

public class FSCheckpointSerializer implements CheckpointSerializer {

    private ConfigManager configManager = ConfigManager.getInstance();
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public void saveCheckpoint(Checkpoint checkpoint, Map<String, String> versionMarPath)
            throws IOException, ConflictStatusException {

        String checkpointPath = configManager.getCheckpointStore() + "/" + checkpoint.getName();
        File checkPointModelStore = new File(checkpointPath + "/model_store");
        if (checkPointModelStore.exists()) {
            throw new ConflictStatusException(
                    "Checkpoint " + checkpoint.getName() + " already exists.");
        }

        boolean checkpointDirectoryCreated = checkPointModelStore.mkdirs();

        if (checkpointDirectoryCreated) {
            for (Map.Entry<String, String> marPath : versionMarPath.entrySet()) {
                String destMarFile = checkPointModelStore + "/" + marPath.getKey() + ".mar";
                FileOutputStream fos = new FileOutputStream(destMarFile);
                ZipOutputStream zos = new ZipOutputStream(fos);
                String modelFilesPath = marPath.getValue();
                File[] marFiles = new File(modelFilesPath).listFiles();
                if (marFiles != null) {
                    for (File file : marFiles) {
                        addDirToZipArchive(zos, file, null);
                    }
                }
                zos.flush();
                fos.flush();
                zos.close();
                fos.close();
                File modelMarFile = new File(destMarFile);
                ZipFile zipFile = new ZipFile(modelMarFile);
                if (zipFile.size() == 0) {
                    if (modelMarFile.delete()) {
                        int i = marPath.getKey().lastIndexOf('_');
                        String[] modelVersion = {
                            marPath.getKey().substring(0, i), marPath.getKey().substring(i + 1)
                        };
                        checkpoint.getModels().get(modelVersion[0]).remove(modelVersion[1]);
                        checkpoint.setModelCount(checkpoint.getModelCount() - 1);
                        if (checkpoint.getModels().get(modelVersion[0]).isEmpty()) {
                            checkpoint.getModels().remove(modelVersion[0]);
                        }
                    }
                }
                zipFile.close();
            }

            String chkpntJson = GSON.toJson(checkpoint, Checkpoint.class);

            try (OutputStream os =
                    new FileOutputStream(checkpointPath + "/" + checkpoint.getName() + ".json")) {
                OutputStreamWriter osWriter = new OutputStreamWriter(os, StandardCharsets.UTF_8);
                osWriter.write(chkpntJson);
                osWriter.flush();
                osWriter.close();
            }
        }
    }

    public Checkpoint getCheckpoint(String checkpointName) throws IOException {
        JsonParser jsonParser = new JsonParser();

        JsonObject checkpointJson = null;

        try (InputStream is =
                new FileInputStream(
                        configManager.getCheckpointStore()
                                + "/"
                                + checkpointName
                                + "/"
                                + checkpointName
                                + ".json")) {

            InputStreamReader isReader = new InputStreamReader(is, StandardCharsets.UTF_8);
            checkpointJson = jsonParser.parse(isReader).getAsJsonObject();
            Checkpoint checkpoint = GSON.fromJson(checkpointJson, Checkpoint.class);
            isReader.close();
            return checkpoint;
        }
    }

    public List<Checkpoint> getAllCheckpoints() throws IOException {

        ArrayList<Checkpoint> resp = new ArrayList<Checkpoint>();

        String[] checkPoints = new File(configManager.getCheckpointStore()).list();
        if (checkPoints != null) {
            for (String checkPointName : checkPoints) {
                if (new File(configManager.getCheckpointStore() + "/" + checkPointName)
                        .isDirectory()) {
                    resp.add(getCheckpoint(checkPointName));
                }
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
                File[] marFiles = fileToZip.listFiles();
                if (marFiles != null) {
                    for (File file : marFiles) {
                        addDirToZipArchive(zos, file, zipEntryName);
                    }
                }
            }
        } else {
            byte[] buffer = new byte[1024];
            FileInputStream fis = null;
            try {
                fis = new FileInputStream(fileToZip);
                zos.putNextEntry(new ZipEntry(zipEntryName));
                int length;
                while ((length = fis.read(buffer)) > 0) {
                    zos.write(buffer, 0, length);
                }
            } finally {
                if (fis != null) {
                    fis.close();
                }
                zos.closeEntry();
            }
        }
    }
}
