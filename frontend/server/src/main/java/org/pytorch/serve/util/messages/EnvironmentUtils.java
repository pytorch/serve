package org.pytorch.serve.util.messages;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;
import org.pytorch.serve.archive.model.Manifest;
import org.pytorch.serve.util.ConfigManager;
import org.pytorch.serve.wlm.Model;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class EnvironmentUtils {
    private static final Logger logger = LoggerFactory.getLogger(EnvironmentUtils.class);
    private static ConfigManager configManager = ConfigManager.getInstance();

    private EnvironmentUtils() {}

    public static String[] getEnvString(String cwd, String modelPath, String handler) {
        ArrayList<String> envList = new ArrayList<>();
        StringBuilder pythonPath = new StringBuilder();
        Pattern blackList = configManager.getBlacklistPattern();

        if (handler != null && handler.contains(":")) {
            String handlerFile = handler;
            handlerFile = handler.split(":")[0];
            if (handlerFile.contains("/")) {
                handlerFile = handlerFile.substring(0, handlerFile.lastIndexOf('/'));
            }

            pythonPath.append(handlerFile).append(File.pathSeparatorChar);
        }

        HashMap<String, String> environment = new HashMap<>(System.getenv());
        environment.putAll(configManager.getBackendConfiguration());

        if (System.getenv("PYTHONPATH") != null) {
            pythonPath.append(System.getenv("PYTHONPATH")).append(File.pathSeparatorChar);
        }

        if (modelPath != null) {
            File modelPathCanonical = new File(modelPath);
            try {
                modelPathCanonical = modelPathCanonical.getCanonicalFile();
            } catch (IOException e) {
                logger.error("Invalid model path {}", modelPath, e);
            }
            pythonPath.append(modelPathCanonical.getAbsolutePath()).append(File.pathSeparatorChar);
            File dependencyPath = new File(modelPath);
            if (Files.isSymbolicLink(dependencyPath.toPath())) {
                pythonPath
                        .append(dependencyPath.getParentFile().getAbsolutePath())
                        .append(File.pathSeparatorChar);
            }
        }

        if (!cwd.contains("site-packages") && !cwd.contains("dist-packages")) {
            pythonPath.append(cwd);
        }

        environment.put("PYTHONPATH", pythonPath.toString());

        for (Map.Entry<String, String> entry : environment.entrySet()) {
            if (!blackList.matcher(entry.getKey()).matches()) {
                envList.add(entry.getKey() + '=' + entry.getValue());
            }
        }

        return envList.toArray(new String[0]); // NOPMD
    }

    public static String getPythonRunTime(Model model) {
        String pythonRuntime;
        Manifest.RuntimeType runtime = model.getRuntimeType();
        if (runtime == Manifest.RuntimeType.PYTHON) {
            pythonRuntime = configManager.getPythonExecutable();
            Path pythonVenvRuntime =
                    Paths.get(getPythonVenvPath(model).toString(), "bin", "python");
            if (model.isUseVenv() && Files.exists(pythonVenvRuntime)) {
                pythonRuntime = pythonVenvRuntime.toString();
            }
        } else {
            pythonRuntime = runtime.getValue();
        }
        return pythonRuntime;
    }

    public static File getPythonVenvPath(Model model) {
        File modelDir = model.getModelDir();
        if (Files.isSymbolicLink(modelDir.toPath())) {
            modelDir = modelDir.getParentFile();
        }
        Path venvPath = Paths.get(modelDir.getAbsolutePath(), "venv").toAbsolutePath();
        return venvPath.toFile();
    }

    public static String[] getCppEnvString(String libPath) {
        ArrayList<String> envList = new ArrayList<>();
        StringBuilder cppPath = new StringBuilder();
        Pattern blackList = configManager.getBlacklistPattern();

        HashMap<String, String> environment = new HashMap<>(System.getenv());
        environment.putAll(configManager.getBackendConfiguration());

        cppPath.append(libPath);

        String os = System.getProperty("os.name").toLowerCase();
        if (os.indexOf("win") >= 0) {
            if (System.getenv("PATH") != null) {
                cppPath.append(File.pathSeparatorChar).append(System.getenv("PATH"));
            }
            environment.put("PATH", cppPath.toString());
        } else if (os.indexOf("mac") >= 0) {
            if (System.getenv("DYLD_LIBRARY_PATH") != null) {
                cppPath.append(File.pathSeparatorChar).append(System.getenv("DYLD_LIBRARY_PATH"));
            }
            environment.put("DYLD_LIBRARY_PATH", cppPath.toString());
        } else {
            if (System.getenv("LD_LIBRARY_PATH") != null) {
                cppPath.append(File.pathSeparatorChar).append(System.getenv("LD_LIBRARY_PATH"));
            }
            environment.put("LD_LIBRARY_PATH", cppPath.toString());
        }

        for (Map.Entry<String, String> entry : environment.entrySet()) {
            if (!blackList.matcher(entry.getKey()).matches()) {
                envList.add(entry.getKey() + '=' + entry.getValue());
            }
        }

        return envList.toArray(new String[0]); // NOPMD
    }
}
