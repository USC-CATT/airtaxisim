FROM ros:noetic

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"

ARG PYVER=3.10.19
RUN cat /etc/os-release


RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential make \
    zlib1g-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev liblzma-dev libncursesw5-dev tk-dev uuid-dev xz-utils \
    libgdbm-dev libgdbm-compat-dev \
 && rm -rf /var/lib/apt/lists/*

#Install Python 3.10 and set it as default
# RUN add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && apt-get install -y python3 python3-distutils python3-venv

#Install Python 3.10 and set it as default(newer method without deadsnakes)
WORKDIR /tmp
RUN curl -fL "https://www.python.org/ftp/python/${PYVER}/Python-${PYVER}.tgz" -o Python.tgz \
 && tar -xzf Python.tgz \
 && cd "Python-${PYVER}" \
 && ./configure --prefix="/opt/python-${PYVER}" --enable-optimizations --with-ensurepip=install \
 && make -j"$(nproc)" \
 && make install \
 && cd / \
 && rm -rf /tmp/Python* /tmp/Python.tgz

ENV PATH="/opt/python-${PYVER}/bin:${PATH}"
ENV VENV_PATH="/venv"
RUN python3.10 -m venv "${VENV_PATH}" \
 && "${VENV_PATH}/bin/python" -m pip install -U pip setuptools wheel

ENV PATH="${VENV_PATH}/bin:${PATH}"

RUN python --version
RUN echo $?

#Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3

#Install NumPy and loguru
RUN python3 -m pip install --upgrade pip
RUN pip3 install numpy loguru 

COPY jax_guam /jax_guam
WORKDIR /jax_guam
RUN python3 -m pip install -e .
RUN pip install jaxlib

# Set up ROS environment
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/noetic
ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
ENV PATH=$ROS_ROOT/bin:$PATH
ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH

RUN mkdir -p /catkin_ws/src
