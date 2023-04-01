package org.pytorch.serve.archive.utils;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.UUID;
import java.util.zip.GZIPInputStream;
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
            md = MessageDigest.getInstance("SHA1");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
        if (isMar) {
            unzip(new DigestInputStream(is, md), tmp);
        } else {
            decompressTarGzipFile(new DigestInputStream(is, md), tmp.toPath());
        }
        if (eTag == null) {
            eTag = UUID.randomUUID().toString().replaceAll("-", "");
        }

        File dir = new File(modelDir, eTag);

        FileUtils.moveDirectory(tmp, dir);

        return dir;
    }

    public static boolean isGZipped(InputStream in) {
        if (!in.markSupported()) {
            in = new BufferedInputStream(in);
        }
        in.mark(2);
        int magic = 0;
        try {
            magic = in.read() & 0xff | ((in.read() << 8) & 0xff00);
            in.reset();
        } catch (IOException e) {
            e.printStackTrace(System.err);
            return false;
        }
        return magic == GZIPInputStream.GZIP_MAGIC;
    }

    public static void decompressTarGzipFile(InputStream is, Path target) throws IOException {
        try (GzipCompressorInputStream gzi = new GzipCompressorInputStream(is);
                TarArchiveInputStream ti = new TarArchiveInputStream(gzi)) {
            ArchiveEntry entry;
            while ((entry = ti.getNextEntry()) != null) {

                // create a new path, zip slip validate
                Path newPath = zipSlipProtect(entry, target);

                if (entry.isDirectory()) {
                    Files.createDirectories(newPath);
                } else {

                    // check parent folder again
                    Path parent = newPath.getParent();
                    if (parent != null) {
                        if (Files.notExists(parent)) {
                            Files.createDirectories(parent);
                        }
                    }

                    // copy TarArchiveInputStream to Path newPath
                    Files.copy(ti, newPath, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }

    private static Path zipSlipProtect(ArchiveEntry entry, Path targetDir) throws IOException {

        Path targetDirResolved = targetDir.resolve(entry.getName());

        // make sure normalized file still has targetDir as its prefix,
        // else throws exception
        Path normalizePath = targetDirResolved.normalize();

        if (!normalizePath.startsWith(targetDir)) {
            throw new IOException("Bad entry: " + entry.getName());
        }

        return normalizePath;
    }
}
