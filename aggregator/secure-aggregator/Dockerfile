FROM Ubuntu

RUN apt-get update
RUN apt-get install python

COPY . aggregator/secure-aggregator/src

CMD ["python", "aggregator/secure-aggregator/src/main.py"]

