FROM ubuntu:20.04

WORKDIR /trainer/MONAI

RUN apt-get update && apt-get install -y \
    software-properties-common

RUN add-apt-repository universe

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip

RUN pip3 install --no-cache-dir numpy

RUN pip3 install --no-cache-dir matplotlib

RUN pip3 install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install --no-cache-dir -q "monai[gdown, nibabel, tqdm, itk]"

COPY . /trainer/MONAI

CMD ["python3", "pipeline/monai-example.py"]