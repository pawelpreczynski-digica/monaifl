FROM Ubuntu

RUN apt-get update
RUN apt-get install python

COPY . data-provider/third-party-dataset/src

CMD ["python", "/data-provider/third-party-dataset/src/main.py"]

