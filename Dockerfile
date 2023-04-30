FROM ubuntu:latest

# Install system packages
RUN apt-get update \
    && apt-get install -y \
        openssh-server  \
        sudo \
        ssh \
    \
        python3-pip \
        python3-dev \
        python3-venv \
        libevent-dev \
    \
        vim \
        neovim \
    \
        gcc \
        g++ \
        gdb \
        clang \
        cmake \
        make \
        build-essential \
        ninja-build \
        autoconf \
        automake \
        valgrind \
    \
        locales-all \
        dos2unix \
    \
        rsync \
        tar \
        doxygen \
        tree \
        wget \
        fish \
    \
        swig \
        software-properties-common \
    && apt-get clean

# Prepare project for running
COPY requirements_for_workflows.txt /home/ubuntu/
RUN python3 -m venv /home/ubuntu/venv \
    && /home/ubuntu/venv/bin/pip install --upgrade pip  \
    && /home/ubuntu/venv/bin/pip install -r /home/ubuntu/requirements_for_workflows.txt

# Setup SSH: username=user:password=user
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 user
RUN  echo 'user:user' | chpasswd
RUN service ssh start

# Port to expose and Entrypoint
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]



## Build Ubuntu image with base functionality.
#FROM ubuntu:20.04 AS ubuntu-base
#ENV DEBIAN_FRONTEND "noninteractive"
#SHELL ["/bin/bash", "-o", "pipefail", "-c"]
#
## Setup the default user.
#RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo ubuntu
#RUN echo 'ubuntu:ubuntu' | chpasswd
#USER ubuntu
#WORKDIR /home/ubuntu
#
## Build image with Python and SSHD.
#FROM ubuntu-base AS ubuntu-with-sshd
#USER root
#
## Install required tools.
#RUN apt-get -qq update \
#    && apt-get -qq --no-install-recommends install vim-tiny=2:8.1.* \
#    && apt-get -qq --no-install-recommends install sudo=1.8.* \
#    && apt-get -qq --no-install-recommends install python3-pip=20.0.* \
#    && apt-get -qq --no-install-recommends install openssh-server=1:8.* \
#    && apt-get -qq clean    \
#    && rm -rf /var/lib/apt/lists/*
#
## My packages
#
##RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
##    tar -xzf ta-lib-0.4.0-src.tar.gz \
##    cd ta-lib \
##    ./configure \
##    make \
##    make install \
##
##RUN add-apt-repository ppa:deadsnakes/ppa && apt install python3.10
#
#
## Configure SSHD.
## SSH login fix. Otherwise user is kicked off after login
##RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
##    && mkdir /var/run/sshd \
##    && bash -c 'install -m755 <(printf "#!/bin/sh\nexit 0") /usr/sbin/policy-rc.d' \
##    && ex +'%s/^#\zeListenAddress/\1/g' -scwq /etc/ssh/sshd_config \
##    && ex +'%s/^#\zeHostKey .*ssh_host_.*_key/\1/g' -scwq /etc/ssh/sshd_config \
##    && RUNLEVEL=1 dpkg-reconfigure openssh-server \
##    && ssh-keygen -A -v \
##    && update-rc.d ssh defaults
##
### Configure sudo.
##RUN ex +"%s/^%sudo.*$/%sudo ALL=(ALL:ALL) NOPASSWD:ALL/g" -scwq! /etc/sudoers
#
## Generate and configure user keys.
#USER ubuntu
#RUN ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
##COPY --chown=ubuntu:root "./files/authorized_keys" /home/ubuntu/.ssh/authorized_keys
#
## Setup default command and/or parameters.
#EXPOSE 22
#CMD ["/usr/bin/sudo", "/usr/sbin/sshd", "-D", "-o", "ListenAddress=0.0.0.0"]
