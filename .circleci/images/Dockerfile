ARG BASE_IMAGE=ubuntu:18.04
ARG ENV_TYPE=python
FROM ${BASE_IMAGE} AS compile-image

ENV PYTHONUNBUFFERED TRUE

# Install JAVA 11 and Node 14
RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        apache2-utils \
        build-essential \
        ca-certificates\
        curl \
        git \
        openjdk-11-jdk \
    && curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        nodejs

# Install python 3.9 using conda env
FROM compile-image as conda39-image
ENV PATH ~/.local/bin/:$PATH
ENV PATH=/miniconda/bin:${PATH}
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh \
        && bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -p /miniconda -b \
        && rm Miniconda3-py39_4.9.2-Linux-x86_64.sh \
        && rm -rf /var/lib/apt/lists/* \
        && conda install -y conda-build

# Install python 3.8 using conda env
FROM compile-image as conda38-image
ENV PATH ~/.local/bin/:$PATH
ENV PATH=/miniconda/bin:${PATH}
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        && bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b \
        && rm Miniconda3-latest-Linux-x86_64.sh \
        && rm -rf /var/lib/apt/lists/* \
        && conda install -y conda-build

# Install vanilla python 3.6
FROM compile-image as pythn36-image
ENV PATH ~/.local/bin/:$PATH
RUN apt-get install --no-install-recommends -y \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && pip3 install --upgrade pip 

# Install python 3.7 using pyenv env
FROM compile-image as pyenv37-image
ENV PATH ~/.local/bin/:$PATH
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}
RUN apt-get install --no-install-recommends -y  libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libxml2-dev libxmlsec1-dev libffi-dev\
    && curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash \
    && pyenv install 3.7.0 \
    && pyenv global 3.7.0 \
    && pip3 install --upgrade pip

# Install python 3.6 and venv env
FROM compile-image as venv36-image
ENV PATH ~/.local/bin/:$PATH
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
RUN apt-get install --no-install-recommends -y \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-venv\
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1\
    && python3 -m venv $VIRTUAL_ENV \
    && pip3 install --upgrade pip


FROM ${ENV_TYPE}-image AS final-image
ARG ENV_TYPE
# Install jmeter for Benchmark
RUN cd /opt \
    && curl https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.3.tgz | tar -xz\
    && cd apache-jmeter-5.3 \
    && ln -s /opt/apache-jmeter-5.3/bin/jmeter /usr/local/bin/jmeter \
    && curl --location -o lib/ext/jmeter-plugins-manager-1.4.jar https://jmeter-plugins.org/get/ \
    && curl --location -o lib/cmdrunner-2.2.jar http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.2/cmdrunner-2.2.jar \
    && java -cp lib/ext/jmeter-plugins-manager-1.4.jar org.jmeterplugins.repository.PluginManagerCMDInstaller \
    && bin/PluginsManagerCMD.sh install jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1

# Install Node dependencies for api-tests and link-checker
RUN npm install -g \
        newman \
        newman-reporter-html \
        markdown-link-check

# Install Torchserve, Pytest, Regression and Benchmark Dependencies
RUN pip install --no-cache-dir \
        Click \
        click-config-file \
        future \
        matplotlib \
        mock \
        pandas \
        pillow \
        psutil \
        pylint \
        pytest \
        pytest-cov \
        pytest-mock \
        requests \
        sentencepiece \
        torch==1.8.1 \
        torchaudio==0.8.1 \
        torchtext==0.9.1 \
        torchvision==0.9.1 \
        transformers==4.6.0 \
        wheel
