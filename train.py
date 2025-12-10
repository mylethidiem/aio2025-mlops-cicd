"""
YOLO Training Script for Defect Detection

This script trains a YOLO model on the defect detection dataset.
It integrates with MLflow for experiment tracking and model management.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union

import torch
import mlflow
from ultralytics import YOLO
from ultralytics.utils.callbacks import mlflow as mlflow_cb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)


def train_yolo(
    data_yaml: str = "data/data.yaml",
    model_name: str = "yolov8n",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    patience: int = 20,
    device: Optional[Union[int, str]] = None,
    experiment_name: str = "yolo-training",
    run_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train YOLO model on the defect detection dataset.
    Supports YOLOv8 and YOLOv11 (requires ultralytics>=8.3.0).
    
    Args:
        data_yaml: Path to data.yaml configuration file
        model_name: YOLO model variant (yolov8n, yolov8s, yolov11n, yolov11s, etc.)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        patience: Early stopping patience
        device: Device to use - int for CUDA GPU ID, 'mps' for Apple Silicon, 'cpu', or None for auto-select (CUDA > MPS > CPU)
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        **kwargs: Additional arguments to pass to YOLO.train()
        
    Returns:
        Dictionary containing training results
    """
    
    logger.info("=" * 50)
    logger.info("YOLO Training Script")
    logger.info("=" * 50)
    
    # Validate data.yaml exists
    data_path = PROJECT_ROOT / data_yaml
    if not data_path.exists():
        raise FileNotFoundError(f"Data configuration file not found: {data_path}")
    
    logger.info(f"Data YAML: {data_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {imgsz}")
    
    # Check device availability (CUDA GPU > MPS > CPU)
    if device is None:
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Determine device name for logging
    if isinstance(device, int):
        device_name = f"CUDA GPU {device}"
    elif device == "mps":
        device_name = "MPS (Apple Silicon)"
    else:
        device_name = device.upper()
    
    logger.info(f"Device: {device_name}")
    
    # Generate run name if not provided
    if run_name is None:
        run_name = f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Configure MLflow for tracking
    # Use file-based backend by default to avoid SQLite concurrency issues
    # For production, consider using a remote MLflow server with PostgreSQL
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    if experiment_name:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    if run_name:
        os.environ["MLFLOW_RUN_NAME"] = run_name
    
    logger.info(f"MLflow tracking URI: {mlflow_uri}")
    logger.info(f"MLflow experiment: {experiment_name}")
    logger.info(f"MLflow run name: {run_name}")
    
    try:
        # Check ultralytics version
        import ultralytics
        ul_version = ultralytics.__version__
        logger.info(f"Ultralytics version: {ul_version}")

        # Initialize YOLO model
        logger.info(f"Loading {model_name} model...")
        logger.info("Note: If this is the first time, the model will be downloaded automatically...")
        model_path = MODELS_DIR / f"{model_name}.pt"
        model = YOLO(model_path)

        # Attach Ultralytics' built-in MLflow callbacks for automatic logging
        for event_name, callback in mlflow_cb.callbacks.items():
            model.add_callback(event_name, callback)
        
        # Train the model
        logger.info("Starting training...")
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            device=device,
            project=str(RUNS_DIR),
            name=run_name,
            save=True,
            cache=False,
            plots=True,
            **kwargs
        )
        
        logger.info("Training completed successfully!")
        
        # Save the best model
        best_model_path = MODELS_DIR / f"{model_name}-best-{run_name}.pt"
        model.save(str(best_model_path))
        logger.info(f"Best model saved to: {best_model_path}")
        
        return {
            "status": "success",
            "run_name": run_name,
            "model_path": str(best_model_path),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def validate_model(
    model_path: str,
    data_yaml: str = "data/data.yaml",
    imgsz: int = 640,
    device: Optional[Union[int, str]] = None,
    split: str = "val",
) -> Dict[str, Any]:
    """
    Validate a trained model on a specific dataset split.
    
    Args:
        model_path: Path to the trained model
        data_yaml: Path to data.yaml configuration file
        imgsz: Input image size
        device: GPU device ID, 'mps', 'cpu', or None for auto-select
        split: Dataset split to validate on ('val' or 'test')
        
    Returns:
        Dictionary with validation metrics
    """
    
    logger.info(f"Validating model on {split} set: {model_path}")
    
    # Load model
    model = YOLO(model_path)
    
    # Validate
    results = model.val(
        data=str(PROJECT_ROOT / data_yaml),
        imgsz=imgsz,
        device=device,
        split=split
    )
    
    # Extract key metrics
    metrics = {
        "precision": float(results.box.p.mean()) if hasattr(results.box, 'p') else 0.0,
        "recall": float(results.box.r.mean()) if hasattr(results.box, 'r') else 0.0,
        "mAP50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
        "mAP50-95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
    }
    
    logger.info(f"{split.capitalize()} set metrics:")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  mAP@50: {metrics['mAP50']:.4f}")
    logger.info(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
    
    return metrics


def _trigger_deployment_if_configured(
    model_name: str,
    model_version: int,
    test_metrics: Dict[str, float]
) -> None:
    """
    Trigger deployment workflows if configured via environment variables.
    
    Args:
        model_name: Name of the registered model
        model_version: Version number of the model
        test_metrics: Dictionary of test set metrics
    """
    
    # Check if deployment trigger is configured
    trigger_deployment = os.getenv("TRIGGER_DEPLOYMENT", "false").lower() == "true"
    
    if not trigger_deployment:
        logger.info("Deployment trigger not configured (set TRIGGER_DEPLOYMENT=true to enable)")
        return
    
    logger.info("\n" + "=" * 50)
    logger.info("Triggering Deployment")
    logger.info("=" * 50)
    
    try:
        # Set environment variables for the trigger script
        os.environ["MODEL_NAME"] = model_name
        os.environ["MODEL_VERSION"] = str(model_version)
        os.environ["TEST_PRECISION"] = str(test_metrics["precision"])
        os.environ["TEST_RECALL"] = str(test_metrics["recall"])
        os.environ["TEST_MAP50"] = str(test_metrics["mAP50"])
        os.environ["TEST_MAP50_95"] = str(test_metrics["mAP50-95"])
        
        # Run the deployment trigger script
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "trigger_deployment.py")],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logger.info("✅ Deployment trigger successful")
            if result.stdout:
                logger.info(result.stdout)
        else:
            logger.warning(f"⚠️ Deployment trigger failed with code {result.returncode}")
            if result.stderr:
                logger.warning(result.stderr)
                
    except Exception as e:
        logger.error(f"❌ Failed to trigger deployment: {e}")


def register_model_to_mlflow(
    model_path: str,
    model_name: str,
    test_metrics: Dict[str, float],
    run_name: str,
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Register the trained model to MLflow Model Registry with test metrics.
    
    Args:
        model_path: Path to the trained model file
        model_name: Name for the registered model
        test_metrics: Dictionary of test set metrics
        run_name: MLflow run name
        experiment_name: MLflow experiment name
        
    Returns:
        Registered model URI
    """
    logger.info("Registering model to MLflow...")
    
    # Set MLflow tracking URI and experiment
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    # Get MLflow client for model registry operations
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    # Start a new MLflow run for model registration
    with mlflow.start_run(run_name=f"{run_name}-registration") as run:
        
        # Log test metrics
        mlflow.log_metrics({
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_mAP50": test_metrics["mAP50"],
            "test_mAP50-95": test_metrics["mAP50-95"],
        })
        
        # Log model parameters
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("model_name", model_name)
        
        # Log the model file as artifact
        mlflow.log_artifact(model_path, artifact_path="model")
        
        # Get the run ID and artifact URI
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
    
    # Ensure the registered model exists
    try:
        client.get_registered_model(model_name)
        logger.info(f"Registered model '{model_name}' already exists.")
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)
        logger.info(f"Created new registered model '{model_name}'.")
    
    # Create a new model version using the artifact URI
    # The source should point to the artifact directory containing the model
    model_filename = os.path.basename(model_path)
    source_uri = f"{artifact_uri}/model/{model_filename}"
    
    registered_model = client.create_model_version(
        name=model_name,
        source=source_uri,
        run_id=run_id,
        description=f"YOLO model with test mAP@50-95: {test_metrics['mAP50-95']:.4f}"
    )
    
    logger.info(f"Model registered successfully!")
    logger.info(f"  Model Name: {registered_model.name}")
    logger.info(f"  Model Version: {registered_model.version}")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Source URI: {source_uri}")
    
    # Check if this is the best model based on test mAP50-95
    current_map = test_metrics["mAP50-95"]
    should_promote = True
    
    # Get all versions of this model
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        if len(all_versions) > 1:  # More than just the current version
            logger.info("Comparing with previous model versions...")
            best_previous_map = 0.0
            
            for version in all_versions:
                if version.version != str(registered_model.version):
                    # Get the run associated with this version
                    try:
                        version_run = client.get_run(version.run_id)
                        version_map = version_run.data.metrics.get("test_mAP50-95", 0.0)
                        if version_map > best_previous_map:
                            best_previous_map = version_map
                    except Exception as e:
                        logger.warning(f"Could not retrieve metrics for version {version.version}: {e}")
            
            if current_map <= best_previous_map:
                should_promote = False
                logger.info(f"Current mAP@50-95: {current_map:.4f} <= Best previous: {best_previous_map:.4f}")
                logger.info("This model will NOT be promoted to 'production'")
            else:
                logger.info(f"Current mAP@50-95: {current_map:.4f} > Best previous: {best_previous_map:.4f}")
                logger.info("This model will be promoted to 'production'")
    except Exception as e:
        logger.warning(f"Could not compare with previous versions: {e}")
        logger.info("Promoting this model to 'production' by default")
    
    # Set "production" alias if this is the best model
    if should_promote:
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=registered_model.version
        )
        logger.info(f"✓ Model version {registered_model.version} set as 'production' alias")
        
        # Trigger deployment if configured
        _trigger_deployment_if_configured(
            model_name=model_name,
            model_version=registered_model.version,
            test_metrics=test_metrics
        )
    else:
        logger.info(f"Model version {registered_model.version} registered but not promoted")
    
    return {
        "model_uri": f"models:/{model_name}/{registered_model.version}",
        "promoted": should_promote,
        "version": registered_model.version
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv11 for defect detection")
    
    # Training arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Path to data.yaml configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n",
        help="YOLO model variant (yolo11n, yolo11s, etc.)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: GPU ID (e.g., 0), 'mps' for Apple Silicon, 'cpu', or None for auto-select"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="yolo-training",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name"
    )
    
    # Validation arguments
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Show validation set metrics (test set evaluation is always performed)"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse device argument (convert numeric strings to int)
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        
        # Train the model (uses train/val sets internally)
        logger.info("Training uses train/val sets to find the best model...")
        result = train_yolo(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            patience=args.patience,
            device=device,
            experiment_name=args.experiment_name,
            run_name=args.run_name
        )
        
        model_path = result["model_path"]
        run_name = result["run_name"]
        
        # Validate on validation set if requested
        if args.validate:
            logger.info("\n" + "=" * 50)
            logger.info("Validation Set Evaluation")
            logger.info("=" * 50)
            validate_model(model_path, args.data, imgsz=args.imgsz, device=device, split="val")
        
        # Always evaluate on test set and register model
        logger.info("\n" + "=" * 50)
        logger.info("Test Set Evaluation")
        logger.info("=" * 50)
        test_metrics = validate_model(
            model_path=model_path,
            data_yaml=args.data,
            imgsz=args.imgsz,
            device=device,
            split="test"
        )
        
        # Register model to MLflow with test metrics
        logger.info("\n" + "=" * 50)
        logger.info("Model Registration")
        logger.info("=" * 50)
        registered_model_name = f"{args.model}-detection"
        registration_result = register_model_to_mlflow(
            model_path=model_path,
            model_name=registered_model_name,
            test_metrics=test_metrics,
            run_name=run_name,
            experiment_name=args.experiment_name
        )
        
        logger.info("\n" + "=" * 50)
        logger.info("Training Pipeline Completed Successfully!")
        logger.info(f"Best model: {model_path}")
        logger.info(f"Registered as: {registered_model_name}")
        logger.info(f"Model URI: {registration_result['model_uri']}")
        if registration_result['promoted']:
            logger.info(f"✅ Model promoted to production (version {registration_result['version']})")
        else:
            logger.info(f"ℹ️ Model registered but not promoted (version {registration_result['version']})")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)
