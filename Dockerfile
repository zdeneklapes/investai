FROM ubuntu:20.04

USER root

WORKDIR /home/user/project

# Avoid user interaction
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata

RUN apt-get update \
  && apt-get install -y ssh \
                        build-essential \
                        vim \
                        neovim \
                        python \
                        python3-pip \
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
                        openssh-server \
    && apt-get clean \
    && ln -s /usr/bin/make /usr/bin/gmake


#    pip install --upgrade pip && \
#    pip install --no-cache-dir -r /app/requirements.txt && \
#    apt-get update --no-install-recommends &&  \
#    apt-get -y install cron vim --no-install-recommends && \
#    rm -rf /var/lib/apt/lists/* && \
#    service cron start

#COPY . .
#COPY requirements.txt start.sh ./
RUN mkdir /run/sshd \
    # sshd
    && ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
#    && ( \
#    echo 'HostKey /etc/ssh/sshd_config_test_clion'; \
#  ) >/etc/ssh/sshd_config \
    # user
    && useradd -m user \
    && yes password | passwd user \
    && usermod -s /bin/bash user \
    # root
    && yes root | passwd root

#RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
#RUN mkdir /var/run/sshd
#RUN bash -c 'install -m755 <(printf "#!/bin/sh\nexit 0") /usr/sbin/policy-rc.d'
#RUN ex +'%s/^#\zeListenAddress/\1/g' -scwq /etc/ssh/sshd_config
#RUN ex +'%s/^#\zeHostKey .*ssh_host_.*_key/\1/g' -scwq /etc/ssh/sshd_config
RUN RUNLEVEL=1 dpkg-reconfigure openssh-server
RUN ssh-keygen -A -v
RUN update-rc.d ssh defaults

USER user

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]



