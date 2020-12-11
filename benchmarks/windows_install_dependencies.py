import sys
sys.path.append("../ts_scripts")
from ts_scripts.shell_utils import download_save, unzip, rm_file, rm_dir
import os
import subprocess
import locale
import shutil
import argparse

def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    enc = locale.getpreferredencoding()
    output = raw_output.decode(enc)
    err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('jmeter_path', help='Path to install jmeter (Eg. "C:\\Program Files")')
    args = parser.parse_args()

    JMETER_INSTALL_PATH = args.jmeter_path
    JMETER_HOME = os.path.join(JMETER_INSTALL_PATH, "apache-jmeter-5.3")

    print("Downloading Jmeter 5.3..")
    download_save("https://downloads.apache.org//jmeter/binaries/apache-jmeter-5.3.zip")
    unzip("apache-jmeter-5.3.zip", JMETER_INSTALL_PATH, "zip")

    print("Downloading plugins manager...")
    download_save("https://jmeter-plugins.org/get", f"{JMETER_HOME}\\lib\\ext\\", "jmeter-plugins-manager-1.3.jar")

    print("Downloading plugins cmdrunner...")
    download_save("http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar",
                f"{JMETER_HOME}\\lib\\", "cmdrunner-2.2.jar")

    print("Downloading plugins jmeter-plugins-standard...")
    download_save("https://jmeter-plugins.org/downloads/file/JMeterPlugins-Standard-1.4.0.zip")
    unzip("JMeterPlugins-Standard-1.4.0.zip", os.getcwd(), "zip")
    shutil.copy("lib\ext\JMeterPlugins-Standard.jar", f"{JMETER_HOME}\\lib\\ext\\")
    rm_file("JMeterPlugins-Standard-1.4.0.zip")
    rm_dir("lib")

    cmd = 'java -cp "{}\\apache-jmeter-5.3\\lib\\ext\\jmeter-plugins-manager-1.3.jar" org.jmeterplugins.repository.PluginManagerCMDInstaller'.format(JMETER_INSTALL_PATH)
    rc, out, _ = run(cmd)
    if rc != 0:
        print("Command execution failed : ", cmd)
    else:
        print(out)

    print("Installing Plugins..")
    cmd_plugins = '"{}\\apache-jmeter-5.3\\bin\\PluginsManagerCMD.bat" install jpgc-synthesis=2.1 jpgc-filterresults=2.1 jpgc-mergeresults=2.1 jpgc-cmd=2.1 jpgc-perfmon=2.1'.format(JMETER_INSTALL_PATH)
    rc, out, _ = run(cmd_plugins)
    if rc != 0:
        print("Command execution failed : ", cmd_plugins)
    else:
        print(out)

    #removing file
    rm_file("apache-jmeter-5.3.zip")