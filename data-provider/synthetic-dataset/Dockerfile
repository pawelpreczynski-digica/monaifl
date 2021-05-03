FROM ubuntu

RUN apt-get update
RUN apt-get install python

COPY . data-provider/synthetic-dataset/src

CMD ["python", "/data-provider/synthetic-dataset/src/main.py"]

