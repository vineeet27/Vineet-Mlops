# Exp5: Dockerized FastAPI Iris Inference Server

A containerized REST API for real-time inference using a Logistic Regression model trained on the Iris dataset.

## Overview

This is a Docker-based version of Exp4, providing:
- Containerized FastAPI application
- Easy deployment and scaling
- Consistent environment across machines
- Health checks and monitoring
- Production-ready setup

## Files

- **main.py** - FastAPI application with model training and prediction endpoints
- **requirements.txt** - Python dependencies
- **Dockerfile** - Docker image configuration
- **docker-compose.yml** - Multi-container orchestration (optional)
- **.dockerignore** - Exclude unnecessary files from Docker build

## Build & Run

### Option 1: Using Docker CLI

**Build the image:**
```bash
docker build -t iris-inference-api .
```

**Run the container:**
```bash
docker run -p 8000:8000 iris-inference-api
```

### Option 2: Using Docker Compose (Recommended)

**Start the service:**
```bash
docker-compose up -d
```

**Stop the service:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f iris-api
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "service": "Iris Inference API"
}
```

### Root Endpoint
```bash
curl http://localhost:8000/
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

Response:
```json
{
  "predicted_class": "setosa",
  "probabilities": [0.98, 0.02, 0.0]
}
```

### Interactive API Documentation
Open in browser: `http://localhost:8000/docs` (Swagger UI)

## Features

- **Model Training**: Automatically trains Logistic Regression on Iris dataset at startup
- **Feature Scaling**: Pipeline includes StandardScaler for normalized inputs
- **Health Checks**: Docker health checks monitor API availability
- **Containerization**: Lightweight Python 3.11 slim image
- **Hot Reload**: Volume mount for development with auto-reload (in docker-compose)
- **Restart Policy**: Automatic restart on failure

## Docker Image Details

- **Base Image**: `python:3.11-slim` (lightweight, ~160MB)
- **Working Directory**: `/app`
- **Exposed Port**: `8000`
- **Environment**: Production-optimized with disabled bytecode caching

## Development

To modify and test:

```bash
# Rebuild image after changes
docker-compose up --build

# View real-time logs
docker-compose logs -f
```

## Performance Notes

- Model training happens on container startup (~2-3 seconds)
- Inference is fast (< 10ms per request)
- Container memory usage: ~500MB
- Stone

## Deploy to Production

For production deployment:

1. **Push to registry:**
   ```bash
   docker tag iris-inference-api myregistry/iris-api:1.0
   docker push myregistry/iris-api:1.0
   ```

2. **Use Kubernetes/Orchestration:**
   - Scale horizontally with multiple replicas
   - Load balance across containers
   - Monitor with health checks

3. **Security:**
   - Use secrets for sensitive data
   - Run as non-root user (can be added to Dockerfile)
   - Scan image for vulnerabilities

## Cleanup

**Remove containers:**
```bash
docker-compose down -v
```

**Remove images:**
```bash
docker rmi iris-inference-api
```
