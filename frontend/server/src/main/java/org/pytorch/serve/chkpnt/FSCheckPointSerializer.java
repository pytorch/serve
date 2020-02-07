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
import java.util.Map.Entry;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;

public class FSCheckPointSerializer implements CheckpointSerializer {

    private ConfigManager configManager = ConfigManager.getInstance();
    public static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public void saveCheckpoint(
            String checkpointName,
            Map<String, Set<Entry<Double, Model>>> models,
            Map<String, String> defaultVersionsMap)
            throws IOException {

        long created = System.currentTimeMillis();
        JsonObject modelCheckpoint = new JsonObject();
        modelCheckpoint.addProperty("created", created);

        String checkpointPath = configManager.getCheckpointStore() + "/" + checkpointName;
        File checkPointModelStore = new File(checkpointPath + "/model_store");
        if (checkPointModelStore.exists()) {
            throw new IOException("Checkpoint already exists");
        }

        checkPointModelStore.mkdirs();

        for (Map.Entry<String, Set<Entry<Double, Model>>> model : models.entrySet()) {
            for (Map.Entry<Double, Model> versionedModels : model.getValue()) {
                Model vmodel = versionedModels.getValue();
                JsonObject modelData = new JsonObject();
                JsonObject modelVersionData = new JsonObject();
                modelVersionData.addProperty(
                        "default",
                        vmodel.getVersion().equals(defaultVersionsMap.get(model.getKey())));
                modelVersionData.addProperty("marName", vmodel.getModelUrl());
                modelVersionData.addProperty("minWorkers", vmodel.getMinWorkers());
                modelVersionData.addProperty("maxWorkers", vmodel.getMaxWorkers());
                modelVersionData.addProperty("batchSize", vmodel.getBatchSize());
                modelVersionData.addProperty("maxBatchDelay", vmodel.getMaxBatchDelay());

                String destMarFile =
                        checkPointModelStore
                                + "/"
                                + model.getKey()
                                + "_"
                                + versionedModels.getKey()
                                + ".mar";
                FileOutputStream fos = new FileOutputStream(destMarFile);
                ZipOutputStream zos = new ZipOutputStream(fos);
                String modelFilesPath = vmodel.getModelDir().getAbsolutePath();
                for (String filename : new File(modelFilesPath).list())
                    addDirToZipArchive(zos, new File(modelFilesPath + "/" + filename), null);
                zos.flush();
                fos.flush();
                zos.close();
                fos.close();

                modelData.add(String.valueOf(versionedModels.getKey()), modelVersionData);
                modelCheckpoint.add(model.getKey(), modelData);
            }
        }

        FileWriter file = new FileWriter(checkpointPath + "/" + checkpointName + ".json");
        file.write(modelCheckpoint.toString());
        file.flush();
        file.close();
    }

    public void saveCheckpoint(Checkpoint chkpnt) throws IOException {
        String chkpntJson = GSON.toJson(chkpnt, Checkpoint.class);
        try (FileWriter file =
                new FileWriter(
                        configManager.getCheckpointStore() + "/" + chkpnt.getName() + ".json")) {
            file.write(chkpntJson);
            file.flush();
        }
    }

    public JsonObject getCheckpoint(String checkpointName) {
        JsonParser jsonParser = new JsonParser();

        JsonObject checkpointJson = null;
        try (FileReader reader =
                new FileReader(
                        configManager.getCheckpointStore() + "/" + checkpointName + ".json")) {
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
                System.out.println("+" + zipEntryName);
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
