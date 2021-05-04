FROM python

WORKDIR /trainer/reporter/Dockerfile

RUN pip install flask
RUN pip install --no-cache-dir numpy
#RUN pip install copy
RUN pip install requests

COPY . .

EXPOSE 5000

CMD ["python", "src/reporter.py"]
