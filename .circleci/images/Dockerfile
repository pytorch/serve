FROM ubuntu:18.04

ENV PYTHONUNBUFFERED TRUE

ENV PATH ~/.local/bin/:$PATH

# Python 3.6.9
# pip 9.0.1 from /usr/lib/python3/dist-packages (python 3.6)
RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        git \
        openjdk-11-jdk \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
    && curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install jmeter for Benchmark
RUN cd /opt \
    && curl https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.3.tgz | tar -xz\
    && cd apache-jmeter-5.3 \
    && ln -s /opt/apache-jmeter-5.3/bin/jmeter /usr/local/bin/jmeter \
    && curl --location -o lib/ext/jmeter-plugins-manager-1.4.jar https://jmeter-plugins.org/get/ \
    && curl --location -o lib/cmdrunner-2.2.jar http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar \
    && java -cp lib/ext/jmeter-plugins-manager-1.4.jar org.jmeterplugins.repository.PluginManagerCMDInstaller \
    && bin/PluginsManagerCMD.sh install jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1

# Install API tests dependencies
RUN npm install -g \
        newman \
        newman-reporter-html \
        markdown-link-check

# Install Torchserve, Pytest, Regression and Benchmark Dependencies
RUN pip install --no-cache-dir \
        future \
        mock \
        pandas \
        pillow \
        psutil \
        pylint==2.4.4 \
        pytest \
        pytest-cov \
        pytest-mock \
        sentencepiece \
        torch \
        torchtext \
        torchvision \
        transformers
