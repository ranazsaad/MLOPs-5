FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=$RUN_ID

WORKDIR /app

RUN echo "Downloading model for RUN_ID: $RUN_ID"

CMD ["bash", "-c", "echo Running model container with RUN_ID=$RUN_ID"]