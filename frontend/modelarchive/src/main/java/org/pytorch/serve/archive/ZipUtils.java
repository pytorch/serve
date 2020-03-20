package org.pytorch.serve.archive;

import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

public final class ZipUtils {

    private ZipUtils() {}

    public static void zip(File src, File dest, boolean includeRootDir) throws IOException {
        int prefix = src.getCanonicalPath().length();
        if (includeRootDir) {
            prefix -= src.getName().length();
        }
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(dest.toPath()))) {
            addToZip(prefix, src, null, zos);
        }
    }

    public static void unzip(File src, File dest) throws IOException {
        unzip(Files.newInputStream(src.toPath()), dest);
    }

    public static void unzip(InputStream is, File dest) throws IOException {
        try (ZipInputStream zis = new ZipInputStream(is)) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                String name = entry.getName();
                File file = new File(dest, name);
                if (entry.isDirectory()) {
                    FileUtils.forceMkdir(file);
                } else {
                    File parentFile = file.getParentFile();
                    FileUtils.forceMkdir(parentFile);
                    try (OutputStream os = Files.newOutputStream(file.toPath())) {
                        IOUtils.copy(zis, os);
                    }
                }
            }
        }
    }

    public static void addToZip(int prefix, File file, FileFilter filter, ZipOutputStream zos)
            throws IOException {
        String name = file.getCanonicalPath().substring(prefix);
        if (name.startsWith("/")) {
            name = name.substring(1);
        }
        if (file.isDirectory()) {
            if (!name.isEmpty()) {
                ZipEntry entry = new ZipEntry(name + '/');
                zos.putNextEntry(entry);
            }
            File[] files = file.listFiles(filter);
            if (files != null) {
                for (File f : files) {
                    addToZip(prefix, f, filter, zos);
                }
            }
        } else if (file.isFile()) {
            ZipEntry entry = new ZipEntry(name);
            zos.putNextEntry(entry);
            try (FileInputStream fis = (FileInputStream) Files.newInputStream(file.toPath())) {
                IOUtils.copy(fis, zos);
            }
        }
    }
}
