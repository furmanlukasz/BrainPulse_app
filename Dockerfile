# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y git
RUN git clone https://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@github.com/furmanlukasz/BrainPulse.git

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 5000

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]