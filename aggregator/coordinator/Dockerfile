FROM Ubuntu

RUN apt-get update
RUN apt-get install python

COPY . /aggregator/coordinator/src

CMD ["python", "/aggregator/coordinator/src/main.py"]

