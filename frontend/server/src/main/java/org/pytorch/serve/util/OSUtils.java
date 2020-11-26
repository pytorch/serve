package org.pytorch.serve.util;

public final class OSUtils {
    private OSUtils() {}

    public static String getKillCmd() {
        String operatingSystem = System.getProperty("os.name").toLowerCase();
        String killCMD;
        if (operatingSystem.indexOf("win") >= 0) {
            killCMD = "taskkill /f /PID %s";
        } else {
            killCMD = "kill -9 %s";
        }
        return killCMD;
    }
}
