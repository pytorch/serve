from scripts.shell_utils import download_save, unzip, rm_file
from subprocess import run


JMETER_HOME="C:\Program Files\\apache-jmeter-5.3"
print("Downloading Jmeter 5.3..")
download_save("https://downloads.apache.org//jmeter/binaries/apache-jmeter-5.3.zip")
unzip("apache-jmeter-5.3.zip","C:\Program Files","zip")

print("Downloading plugins manager...")
download_save("https://jmeter-plugins.org/get", f"{JMETER_HOME}/lib/ext/", "jmeter-plugins-manager-1.3.jar")
download_save("http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar",
              f"{JMETER_HOME}/lib/", "cmdrunner-2.2.jar")

cmd = ["java", "-cp", f"{JMETER_HOME}/lib/ext/jmeter-plugins-manager-1.3.jar",
       "org.jmeterplugins.repository.PluginManagerCMDInstaller"]
run(cmd)

print("Installing Plugins..")
cmd_plugins = [f"{JMETER_HOME}/bin/PluginsManagerCMD.bat", "install", "jpgc-synthesis=2.1", "jpgc-filterresults=2.1",
               "jpgc-mergeresults=2.1", "jpgc-cmd=2.1", "jpgc-perfmon=2.1"]
run(cmd_plugins)
rm_file("apache-jmeter-5.3")
