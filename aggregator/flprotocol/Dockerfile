FROM ubuntu

RUN apt-get update
RUN apt-get install python

COPY . aggregator/flprotocol/src

CMD ["python", "/aggregator/flprotocol/src/main.py"]

