FROM python

WORKDIR /aggregator/coordinator/Dockerfile

RUN pip install flask
RUN pip install --no-cache-dir numpy


COPY . .

EXPOSE 5000

CMD ["python", "src/coordinator.py"]
