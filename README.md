# YOLO Object Detection API with CI/CD

A complete FastAPI backend service for object detection using YOLO model, with full CI/CD pipeline demonstration.

## 🚀 Features

- **FastAPI Backend**: High-performance REST API for object detection
- **YOLO Model**: YOLO11n for real-time object detection
- **Docker Support**: Fully containerized application
- **CI/CD Pipeline**: Complete GitHub Actions workflow
- **Automated Testing**: Unit tests with pytest
- **Health Checks**: Built-in monitoring endpoints

## 📋 API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `GET /model-info` - Get model information and available classes
- `POST /predict` - Upload image and get object detection results

## 🏗️ Project Structure

```
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── tests/
│   ├── __init__.py
│   └── test_main.py         # Unit tests
├── .github/
│   └── workflows/
│       └── ci-cd.yml        # CI/CD pipeline
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Local deployment
├── requirements.txt         # Python dependencies
└── README.md
```

## 🛠️ Local Development

### Prerequisites

- Python 3.10+ (3.11.9 recommended)
- Docker & Docker Compose

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ThuanNaN/aio2025-mlops-cicd.git
   cd aio2025-mlops-cicd
   ```

2. **Install dependencies**
   ```bash
   conda create -n yolo-fastapi python=3.11.9 -y
   conda activate yolo-fastapi

   pip3 install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Using Docker

1. **Build and run with Docker Compose**
   ```bash
   docker compose up --build
   ```

2. **Or use Docker directly**
   ```bash
   docker build -t yolo-fastapi .
   docker run -p 8000:8000 yolo-fastapi
   ```

## 🧪 Testing

Run tests locally:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=app --cov-report=html
```

## 📦 CI/CD Pipeline

The GitHub Actions workflow includes:

### Continuous Integration (CI)
- ✅ Code checkout
- ✅ Python environment setup
- ✅ Dependency installation
- ✅ Linting with flake8
- ✅ Unit tests with pytest
- ✅ Code coverage reporting

### Continuous Deployment (CD)
- 🐳 Docker image build
- 🐳 Push to Docker Hub
- 🐳 Image testing
- 🚀 Deploy to staging (fastapi branch)
- 🚀 Deploy to production (main branch)
- 🔄 Rollback on failure

### Workflow Triggers
- Push to `main` or `fastapi` branches
- Pull requests to `main` branch

## 🔧 Configuration

### GitHub Secrets Required

Add these secrets to your GitHub repository:

- `DOCKER_USERNAME` - Your Docker Hub username
- `DOCKER_PASSWORD` - Your Docker Hub password/token

### Environment Configuration

The pipeline uses two environments:
- **Staging**: Auto-deploys from `fastapi` branch
- **Production**: Auto-deploys from `main` branch

## 📝 Usage Example

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model-info

# Predict objects in image
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### Using Python

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Response Format

```json
{
  "filename": "image.jpg",
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 300.7,
        "y2": 500.2
      }
    }
  ],
  "count": 1
}
```

## 🔄 Deployment Flow

1. **Developer pushes code** to `fastapi` branch
2. **CI Pipeline runs**:
   - Linting
   - Unit tests
   - Code coverage
3. **Build Docker image**
4. **Push to Docker Hub**
5. **Deploy to Staging**
6. **Smoke tests on staging**
7. **Merge to main** (manual approval)
8. **Deploy to Production**

## 📊 Monitoring

The application includes:
- Health check endpoint for load balancers
- Docker health checks
- Logging for all requests
- Error tracking

## 📊 MLflow Tracking

Start MLflow UI server with SQLite backend:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns/artifacts \
  --host 0.0.0.0 \
  --port 5001 \
  --allowed-hosts '*' \
  --cors-allowed-origins '*'
```

Access MLflow UI at: http://localhost:5001

**Note**: The project now uses SQLite database backend instead of filesystem tracking for better features and future compatibility.

# Train first model
python train.py --model yolo11n --epochs 10

# Train another model (only promoted if better)
python train.py --model yolo11n --epochs 20

# Try different model size
python train.py --model yolo11s --epochs 10