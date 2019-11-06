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
import java.io.File;
import java.io.FileFilter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public final class Exporter {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private Exporter() {}

    public static void main(String[] args) {
        String jarName = getJarName();

        Options options = Config.getOptions();
        DefaultParser parser = new DefaultParser();
        try {
            if (args.length == 0
                    || args[0].equalsIgnoreCase("-h")
                    || args[0].equalsIgnoreCase("--help")) {
                printHelp("java -jar " + jarName + " <export>", options);
                return;
            }

            CommandLine cmd = parser.parse(options, args, null, false);
            List<String> cmdArgs = cmd.getArgList();
            if (cmdArgs.isEmpty()) {
                printHelp("java -jar " + jarName + " <export>", options);
                return;
            }
            Config config = new Config(cmd);
            String action = cmdArgs.get(0);
            if (!"export".equalsIgnoreCase(action)) {
                printHelp("java -jar " + jarName + " <export>", options);
                return;
            }

            String modelName = config.getModelName();
            if (!modelName.matches("[A-Za-z][A-Za-z0-9_\\-.]+")) {
                System.err.println(
                        "model-name must starts with letter and only allows alphanumeric characters, hyphens, underscore or dot.");
                return;
            }

            File modelPath = new File(config.getModelPath()).getCanonicalFile();
            if (!modelPath.exists()) {
                System.err.println("model-path not found: " + modelName);
                return;
            }
            String output = config.getOutputFile();
            File outputFile;
            if (output == null) {
                outputFile = new File(modelPath.getParentFile(), modelName + ".mar");
            } else {
                outputFile = new File(output);
            }

            final String fileName = modelPath.getName();
            if (modelPath.isFile() && fileName.endsWith(".model") || fileName.endsWith(".mar")) {
                ModelArchive.migrate(modelPath, outputFile);
                return;
            }

            if (!modelPath.isDirectory()) {
                System.err.println("model-path should be a directory or model archive file.");
                return;
            }

            File[] files = modelPath.listFiles();
            if (files == null) {
                throw new AssertionError(
                        "Failed list files in folder: " + modelPath.getAbsolutePath());
            }

            Manifest manifest = new Manifest();
            Manifest.Model model = new Manifest.Model();
            manifest.setModel(model);

            String runtime = config.getRuntime();
            if (runtime != null) {
                manifest.setRuntime(Manifest.RuntimeType.fromValue(runtime));
            }

            File symbolFile = findUniqueFile(files, "-symbol.json");
            if (symbolFile != null) {
                model.addExtension("symbolFile", symbolFile.getName());
            }

            File paramsFile = findUniqueFile(files, ".params");
            if (paramsFile != null) {
                model.addExtension("parametersFile", paramsFile.getName());
            }

            String handler = config.getHandler();
            if (handler == null) {
                File serviceFile = findUniqueFile(files, "_service.py");
                if (serviceFile != null) {
                    model.setHandler(serviceFile.getName());
                }
            } else {
                Manifest.RuntimeType runtimeType = manifest.getRuntime();
                if (runtimeType == Manifest.RuntimeType.PYTHON
                        || runtimeType == Manifest.RuntimeType.PYTHON2
                        || runtimeType == Manifest.RuntimeType.PYTHON3) {
                    String[] tokens = handler.split(":");
                    File serviceFile = new File(modelPath, tokens[0]);
                    if (serviceFile.exists()) {
                        System.err.println("handler file is not found in: " + modelPath);
                        return;
                    }
                }
                model.setHandler(handler);
            }

            try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(outputFile))) {
                zos.putNextEntry(new ZipEntry("MANIFEST.json"));
                zos.write(GSON.toJson(manifest).getBytes(StandardCharsets.UTF_8));

                int prefix = modelPath.getCanonicalPath().length();

                FileFilter filter =
                        pathname -> {
                            if (pathname.isHidden()) {
                                return false;
                            }
                            String name = pathname.getName();
                            return !"MANIFEST.json".equalsIgnoreCase(name);
                        };

                for (File file : files) {
                    if (filter.accept(file)) {
                        ZipUtils.addToZip(prefix, file, filter, zos);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
                if (!outputFile.delete()) {
                    outputFile.deleteOnExit();
                }
            }
        } catch (InvalidModelException | IOException e) {
            System.err.println(e.getMessage());
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            printHelp("java -jar " + jarName + " <export>", options);
        }
    }

    private static void printHelp(String message, Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(1);
        formatter.setWidth(120);
        formatter.printHelp(message, options);
    }

    private static String getJarName() {
        URL url = Exporter.class.getProtectionDomain().getCodeSource().getLocation();
        String path = url.getPath();
        if ("file".equalsIgnoreCase(url.getProtocol())) {
            File file = new File(path);
            if (path.toLowerCase().endsWith(".jar")) { // we only support jar file for now
                return file.getName();
            }
        }
        return null;
    }

    private static File findUniqueFile(File[] list, String extension) throws InvalidModelException {
        File ret = null;
        for (File file : list) {
            if (file.getName().endsWith(extension)) {
                if (ret != null) {
                    throw new InvalidModelException(
                            "Multiple " + extension + " file found in the path.");
                }
                ret = file;
            }
        }
        return ret;
    }

    private static final class Config {

        private String modelName;
        private String modelPath;
        private String handler;
        private String runtime;
        private String outputFile;

        public Config(CommandLine cmd) {
            modelName = cmd.getOptionValue("model-name");
            modelPath = cmd.getOptionValue("model-path");
            handler = cmd.getOptionValue("handler");
            runtime = cmd.getOptionValue("runtime");
            handler = cmd.getOptionValue("handler");
            outputFile = cmd.getOptionValue("output-file");
        }

        public static Options getOptions() {
            Options options = new Options();
            options.addOption(
                    Option.builder("n")
                            .longOpt("model-name")
                            .hasArg()
                            .required()
                            .argName("MODEL_NAME")
                            .desc(
                                    "Exported model name. Exported file will be named as model-name.model and saved in current working directory.")
                            .build());
            options.addOption(
                    Option.builder("p")
                            .longOpt("model-path")
                            .hasArg()
                            .required()
                            .argName("MODEL_PATH")
                            .desc(
                                    "Path to the folder containing model related files or legacy model archive. Signature file is required.")
                            .build());
            options.addOption(
                    Option.builder("r")
                            .longOpt("runtime")
                            .hasArg()
                            .argName("RUNTIME")
                            .desc(
                                    "The runtime environment for the MMS to execute your model custom code, default python2.7")
                            .build());
            options.addOption(
                    Option.builder("e")
                            .longOpt("engine")
                            .hasArg()
                            .argName("engine")
                            .desc("The ML framework for your model, default MXNet")
                            .build());
            options.addOption(
                    Option.builder("s")
                            .longOpt("handler")
                            .hasArg()
                            .argName("HANDLER")
                            .desc(
                                    "The entry-point within your code that MMS can call to begin execution.")
                            .build());
            options.addOption(
                    Option.builder("o")
                            .longOpt("output-file")
                            .hasArg()
                            .argName("OUTPUT_FILE")
                            .desc("Output model archive file path.")
                            .build());
            return options;
        }

        public String getModelName() {
            return modelName;
        }

        public void setModelName(String modelName) {
            this.modelName = modelName;
        }

        public String getModelPath() {
            return modelPath;
        }

        public void setModelPath(String modelPath) {
            this.modelPath = modelPath;
        }

        public String getHandler() {
            return handler;
        }

        public void setHandler(String handler) {
            this.handler = handler;
        }

        public String getOutputFile() {
            return outputFile;
        }

        public void setOutputFile(String outputFile) {
            this.outputFile = outputFile;
        }

        public String getRuntime() {
            return runtime;
        }

        public void setRuntime(String runtime) {
            this.runtime = runtime;
        }
    }
}
