package org.pytorch.serve.chkpnt;

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
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.pytorch.serve.util.ConfigManager;

public class FSCheckPointSerializer implements CheckpointSerializer {

    private ConfigManager configManager = ConfigManager.getInstance();
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public void saveCheckpoint(Checkpoint chkpnt, Map<String, String> versionMarPath)
            throws IOException {
        String chkpntJson = GSON.toJson(chkpnt, Checkpoint.class);

        String checkpointPath = configManager.getCheckpointStore() + "/" + chkpnt.getName();
        File checkPointModelStore = new File(checkpointPath + "/model_store");
        if (checkPointModelStore.exists()) {
            throw new IOException("Checkpoint already exists");
        }
        checkPointModelStore.mkdirs();

        for (Map.Entry<String, String> marPath : versionMarPath.entrySet()) {
            String destMarFile = checkPointModelStore + "/" + marPath.getKey() + ".mar";
            FileOutputStream fos = new FileOutputStream(destMarFile);
            ZipOutputStream zos = new ZipOutputStream(fos);
            String modelFilesPath = marPath.getValue();
            for (String filename : new File(modelFilesPath).list())
                addDirToZipArchive(zos, new File(modelFilesPath + "/" + filename), null);
            zos.flush();
            fos.flush();
            zos.close();
            fos.close();
        }
        try (FileWriter file = new FileWriter(checkpointPath + "/" + chkpnt.getName() + ".json")) {
            file.write(chkpntJson);
            file.flush();
        }
    }

    public JsonObject getCheckpoint(String checkpointName) {
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

        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return checkpointJson;
    }

    public void removeCheckpoint(String checkpointName) {}

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
