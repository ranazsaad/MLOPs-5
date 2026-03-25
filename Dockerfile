FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=$RUN_ID

WORKDIR /app

# Install MLflow if needed for downloading models
RUN pip install mlflow

# Download model from MLflow (simulation)
RUN echo "Downloading model for RUN_ID: $RUN_ID" && \
    echo "mlflow models download -r $RUN_ID" > download_model.sh && \
    chmod +x download_model.sh

# In a real scenario, you'd download the actual model
# RUN mlflow models download -r $RUN_ID

CMD ["bash", "-c", "echo Running model container with RUN_ID=$RUN_ID && ./download_model.sh"]