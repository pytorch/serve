package org.pytorch.serve.util;

public final class OSUtils {
    private OSUtils() {}

    public static String getKillCmd(long pid) {
        String operatingSystem = System.getProperty("os.name").toLowerCase();
        String killCMD;
        if (operatingSystem.indexOf("win") >= 0) {
            killCMD = "taskkill /f /IM /PID " + pid + " /T";
        } else {
            killCMD = "kill -9 " + pid;
        }
        return killCMD;
    }
}
