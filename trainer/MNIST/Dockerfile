ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:21.02-py3
FROM ${PYTORCH_IMAGE}
LABEL maintainer="monai.contact@gmail.com"
WORKDIR /opt/monai
# install full deps
COPY requirements.txt requirements-min.txt requirements-dev.txt /tmp/
RUN cp /tmp/requirements.txt /tmp/req.bak \
  && awk '!/torch/' /tmp/requirements.txt > /tmp/tmp && mv /tmp/tmp /tmp/requirements.txt \
  && python -m pip install --upgrade --no-cache-dir pip \
  && python -m pip install --no-cache-dir -r /tmp/requirements-dev.txt
# compile ext and remove temp files
# TODO: remark for issue [revise the dockerfile #1276](https://github.com/Project-MONAI/MONAI/issues/1276)
# please specify exact files and folders to be copied -- else, basically always, the Docker build process cannot cache
# this or anything below it and always will build from at most here; one file change leads to no caching from here on...

COPY LICENSE CHANGELOG.md CODE_OF_CONDUCT.md CONTRIBUTING.md README.md versioneer.py setup.py setup.cfg runtests.sh MANIFEST.in ./
#COPY tests ./tests
#COPY monai ./monai

COPY . /trainer/MONAI


RUN BUILD_MONAI=1 FORCE_CUDA=1 python setup.py develop \
  && rm -rf build __pycache__
# NGC Client
WORKDIR /opt/tools
ARG NGC_CLI_URI="https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip"
RUN wget -q ${NGC_CLI_URI} && \
    unzip ngccli_cat_linux.zip && chmod u+x ngc && \
    md5sum -c ngc.md5 && \
    rm -rf ngccli_cat_linux.zip ngc.md5
RUN apt-get update \
  && DEBIAN_FRONTEND="noninteractive" apt-get install -y libopenslide0  \
  && rm -rf /var/lib/apt/lists/*
# append /opt/tools to runtime path for NGC CLI to be accessible from all file system locations
ENV PATH=${PATH}:/opt/tools
WORKDIR /opt/monai








COPY . /trainer/MONAI







CMD ["python3", "pipeline/monai-example.py"]