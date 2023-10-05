package org.pytorch.serve.archive.utils;

import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

public final class ZipUtils {

    private ZipUtils() {}

    public static void unzip(InputStream is, File dest) throws IOException {
        try (ZipInputStream zis = new ZipInputStream(is)) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                File file = new File(dest, entry.getName());
                File canonicalDestDir = dest.getCanonicalFile();
                File canonicalFile = file.getCanonicalFile();

                // Check for Zip Slip vulnerability
                if (!canonicalFile.getPath().startsWith(canonicalDestDir.getPath())) {
                    throw new IOException("Detected Zip Slip vulnerability: " + entry.getName());
                }

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

    public static File unzip(InputStream is, String eTag, String type, boolean isMar)
            throws IOException {
        File tmpDir = FileUtils.getTempDirectory();
        File modelDir = new File(tmpDir, type);
        FileUtils.forceMkdir(modelDir);

        File tmp = File.createTempFile(type, ".download");
        FileUtils.forceDelete(tmp);
        FileUtils.forceMkdir(tmp);

        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
        if (isMar) {
            unzip(new DigestInputStream(is, md), tmp);
        } else {
            decompressTarGzipFile(new DigestInputStream(is, md), tmp);
        }
        if (eTag == null) {
            eTag = UUID.randomUUID().toString().replaceAll("-", "");
        }

        File dir = new File(modelDir, eTag);

        FileUtils.moveDirectory(tmp, dir);

        return dir;
    }

    public static void decompressTarGzipFile(InputStream is, File dest) throws IOException {
        try (GzipCompressorInputStream gzi = new GzipCompressorInputStream(is);
                TarArchiveInputStream tis = new TarArchiveInputStream(gzi)) {
            ArchiveEntry entry;
            while ((entry = tis.getNextEntry()) != null) {
                String name = entry.getName().substring(entry.getName().indexOf('/') + 1);
                File file = new File(dest, name);
                File canonicalDestDir = dest.getCanonicalFile();
                File canonicalFile = file.getCanonicalFile();

                // Check for Zip Slip vulnerability
                if (!canonicalFile.getPath().startsWith(canonicalDestDir.getPath())) {
                    throw new IOException("Detected Zip Slip vulnerability: " + entry.getName());
                }

                if (entry.isDirectory()) {
                    FileUtils.forceMkdir(file);
                } else {
                    File parentFile = file.getParentFile();
                    FileUtils.forceMkdir(parentFile);
                    try (OutputStream os = Files.newOutputStream(file.toPath())) {
                        IOUtils.copy(tis, os);
                    }
                }
            }
        }
    }

    public static File createTempDir(String eTag, String type) throws IOException {
        File tmpDir = FileUtils.getTempDirectory();
        File modelDir = new File(tmpDir, type);

        if (eTag == null) {
            eTag = UUID.randomUUID().toString().replaceAll("-", "");
        }

        File dir = new File(modelDir, eTag);
        if (dir.exists()) {
            FileUtils.forceDelete(dir);
        }
        FileUtils.forceMkdir(dir);

        return dir;
    }

    public static File createSymbolicDir(File source, File dest) throws IOException {
        String sourceDirName = source.getName();
        File targetLink = new File(dest, sourceDirName);
        Files.createSymbolicLink(targetLink.toPath(), source.toPath());

        return targetLink;
    }
}
