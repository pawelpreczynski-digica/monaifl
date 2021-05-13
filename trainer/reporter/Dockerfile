FROM python

WORKDIR /trainer/reporter/Dockerfile

RUN pip install flask
RUN pip install --no-cache-dir numpy
#RUN pip install copy
RUN pip install requests
RUN pip install grpcio
RUN pip install grpcio-tools

COPY . .

#EXPOSE 5000

CMD ["python", "src/client.py"]
