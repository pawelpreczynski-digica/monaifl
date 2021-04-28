FROM Ubuntu

RUN apt-get update
RUN apt-get install python

COPY . data-provider/custom-dataset/src

CMD ["python", "/data-provider/custom-dataset/src/main.py"]

