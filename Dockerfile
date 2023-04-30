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

WORKDIR /home/ubuntu

# Prepare project for running
# Must be in this stupid way, because gym has some problems with pip install -r requirements.txt
# and gymnasium can't be used, because stable-baseline3 is not compatible with gymnasium
COPY requirements.txt /home/ubuntu/
RUN python3 -m venv /home/ubuntu/venv \
    && /home/ubuntu/venv/bin/pip install --upgrade pip  \
    && for i in $(cat requirements.txt);do /home/ubuntu/venv/bin/pip install ${i};done

# Setup SSH: username=user:password=user
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 user
RUN  echo 'user:user' | chpasswd
RUN service ssh start

# Port to expose and Entrypoint
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]