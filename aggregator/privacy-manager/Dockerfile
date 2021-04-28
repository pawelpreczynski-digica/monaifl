FROM Ubuntu

RUN apt-get update
RUN apt-get install python

COPY . aggregator/privacy-manager/src

CMD ["python", "/aggregator/privacy-manager/src/main.py"]

