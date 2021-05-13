FROM python

WORKDIR /aggregator/coordinator/Dockerfile

RUN pip install flask
RUN pip install --no-cache-dir numpy
RUN pip install requests
RUN pip install grpcio
RUN pip install grpcio-tools

COPY . .

#EXPOSE 5000

CMD ["python", "src/server.py"]
