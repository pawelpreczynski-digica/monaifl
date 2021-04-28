FROM Ubuntu

RUN apt-get update
RUN apt-get install python

COPY . trainer/reporter/src

CMD ["python", "/trainer/reporter/src/main.py"]

