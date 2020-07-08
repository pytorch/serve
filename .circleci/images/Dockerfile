FROM ubuntu:18.04

ENV PYTHONUNBUFFERED TRUE

# Python 3.6.9
# pip 9.0.1 from /usr/lib/python3/dist-packages (python 3.6)
RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        build-essential \
        openjdk-11-jdk

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

ENV PATH ~/.local/bin/:$PATH

# Install Torch Dependencies
RUN pip install --no-cache-dir \
        torch \
        torchvision \
        torchtext \
        sentencepiece \
        psutil \
        future

# Install Pytest Dependencies
RUN pip install \
        mock \
        pytest \
        pylint==2.4.4 \
        pytest-mock \
        pytest-cov

# Install Benchmark Dependencies
RUN pip install \
        pillow \
        pandas


# Install NodeJS
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt-get install -y nodejs

# Install API tests dependencies
RUN npm install -g \
        newman \
        newman-reporter-html


# Install jmeter for Benchmark
RUN cd /opt \
    && curl -O https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.3.tgz \
    && tar -xzf apache-jmeter-5.3.tgz \
    && cd apache-jmeter-5.3 \
    && ln -s /opt/apache-jmeter-5.3/bin/jmeter /usr/local/bin/jmeter \
    && curl --location -o lib/ext/jmeter-plugins-manager-1.4.jar https://jmeter-plugins.org/get/ \
    && curl --location -o lib/cmdrunner-2.2.jar http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar \
    && java -cp lib/ext/jmeter-plugins-manager-1.4.jar org.jmeterplugins.repository.PluginManagerCMDInstaller \
    && bin/PluginsManagerCMD.sh install jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1
