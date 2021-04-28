FROM Ubuntu

RUN apt-get update
RUN apt-get install python

COPY . trainer/MONAI/src

CMD ["python", "/trainer/MONAI/src/main.py"]

