FROM ubuntu

RUN apt-get update
RUN apt-get install python

COPY . trainer/Substra/src

CMD ["python", "/trainer/Substra/src/main.py"]

