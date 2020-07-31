FROM python:3.6-slim-buster
ENV PYTHONUNBUFFERED 1
ENV TZ=Europe/Vienna

RUN apt update -y
RUN mkdir /src
WORKDIR /src
COPY requirements.txt /src/
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

COPY ./ /src/

EXPOSE 5000
ENTRYPOINT ["python", "script.py"]