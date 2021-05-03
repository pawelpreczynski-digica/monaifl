FROM python:3.8

# Set temp work directory for package configuration
WORKDIR /usr/src/cache

# Update apt packages
RUN apt update
RUN apt upgrade -y

# Install vim
RUN apt install vim -y

# Install python 3.8
RUN apt install python3.8 -y

# Make python 3.8 the default
RUN echo "alias python=python3.8" >> ~/.bashrc
RUN export PATH=${PATH}:/usr/bin/python3.8
RUN /bin/bash -c "source ~/.bashrc"

# Install pip
RUN apt install python3-pip -y
RUN python3 -m pip install --upgrade pip


COPY . /aggregator/coordinator/src

CMD ["python3", './src/main.py']

