# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y git

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000

COPY . /app

# ENTRYPOINT [ "python" ]
# ENTRYPOINT [ "python" ]
# ENTRYPOINT [ "streamlit run BrainPulseAPP.py" ]

# CMD [ "BrainPulseAPP.py" ]
CMD ["streamlit", "run", "BrainPulseAPP.py"]