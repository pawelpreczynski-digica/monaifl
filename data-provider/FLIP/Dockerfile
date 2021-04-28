FROM Ubuntu

RUN apt-get update
RUN apt-get install python

COPY . data-provider/FLIP/src

CMD ["python", "/data-provider/FLIP/src/main.py"]

