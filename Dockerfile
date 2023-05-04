FROM ubuntu:20.04

# Install system packages
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata
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
    \
        gcc \
        g++ \
        gdb \
        clang \
        cmake \
        make \
        build-essential \
        autoconf \
        automake \
        valgrind \
        software-properties-common \
    \
        rsync \
        tar \
        tree \
        wget \
    && apt-get clean
#        ninja-build \
#        neovim \
#        swig \
#    \
#        locales-all \
#        dos2unix \
#    \
#        doxygen \
#        fish \

WORKDIR /home/ubuntu

# Prepare project for running
# Must be in this stupid way, because gym has some problems with `pip install -r requirements.txt`
# and gymnasium can't be used, because stable-baseline3 is not compatible with gymnasium
COPY requirements.txt /home/ubuntu/
RUN sudo python3 -m venv /home/ubuntu/venv \
    && sudo /home/ubuntu/venv/bin/pip install --upgrade pip \
    && for i in $(cat requirements.txt);do sudo /home/ubuntu/venv/bin/pip install ${i};done

# Setup SSH: username=user:password=user
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 user
RUN echo 'user:user' | chpasswd
RUN service ssh start

# Port to expose and Entrypoint
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]