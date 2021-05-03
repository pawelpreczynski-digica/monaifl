FROM ubuntu

RUN apt-get update
RUN apt-get install python

COPY . trainer/data-manager/src

CMD ["python", "/trainer/data-manager/src/main.py"]

