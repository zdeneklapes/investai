# Build Ubuntu image with base functionality.
FROM ubuntu:20.04 AS ubuntu-base
ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Setup the default user.
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo ubuntu
RUN echo 'ubuntu:ubuntu' | chpasswd
USER ubuntu
WORKDIR /home/ubuntu

# Build image with Python and SSHD.
FROM ubuntu-base AS ubuntu-with-sshd
USER root

# Install required tools.
RUN apt-get -qq update \
    && apt-get -qq --no-install-recommends install vim-tiny=2:8.1.* \
    && apt-get -qq --no-install-recommends install sudo=1.8.* \
    && apt-get -qq --no-install-recommends install python3-pip=20.0.* \
    && apt-get -qq --no-install-recommends install openssh-server=1:8.* \
    && apt-get -qq clean    \
    && rm -rf /var/lib/apt/lists/*

# My packages
RUN apt-get update \
    && apt-get install -y ssh \
                          build-essential \
                          vim \
                          neovim \
                          python \
                          fish \
                          gcc \
                          g++ \
                          gdb \
                          clang \
                          make \
                          cmake \
                          ninja-build \
                          cmake \
                          autoconf \
                          automake \
                          locales-all \
                          dos2unix \
                          rsync \
                          tar \
                          doxygen \
                          valgrind \
                          tree \
    && apt-get clean \
    && ln -s /usr/bin/make /usr/bin/gmake

# Configure SSHD.
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    && mkdir /var/run/sshd \
    && bash -c 'install -m755 <(printf "#!/bin/sh\nexit 0") /usr/sbin/policy-rc.d' \
    && ex +'%s/^#\zeListenAddress/\1/g' -scwq /etc/ssh/sshd_config \
    && ex +'%s/^#\zeHostKey .*ssh_host_.*_key/\1/g' -scwq /etc/ssh/sshd_config \
    && RUNLEVEL=1 dpkg-reconfigure openssh-server \
    && ssh-keygen -A -v \
    && update-rc.d ssh defaults

# Configure sudo.
RUN ex +"%s/^%sudo.*$/%sudo ALL=(ALL:ALL) NOPASSWD:ALL/g" -scwq! /etc/sudoers

# Generate and configure user keys.
USER ubuntu
RUN ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
#COPY --chown=ubuntu:root "./files/authorized_keys" /home/ubuntu/.ssh/authorized_keys

# Setup default command and/or parameters.
EXPOSE 22
CMD ["/usr/bin/sudo", "/usr/sbin/sshd", "-D", "-o", "ListenAddress=0.0.0.0"]
