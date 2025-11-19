from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from src.monitoring.prometheus_metrics import track_inference_time
import sys
from pathlib import Path
import time
import os
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from .auth import verify_token  
from src.models.predictor import CatDogPredictor  

from src.database.db_connector import get_db  
from src.database.feedback_service import FeedbackService  

from src.monitoring.dashboard_service import DashboardService 



ENABLE_PROMETHEUS = os.getenv('ENABLE_PROMETHEUS', 'false').lower() == 'true'


ENABLE_DISCORD = os.getenv('DISCORD_WEBHOOK_URL') is not None

alert_high_latency = None
alert_database_disconnected = None
notifier = None
track_prediction = None
update_db_status = None

if ENABLE_PROMETHEUS:
    try:
        from src.monitoring.prometheus_metrics import (
            update_db_status as _update_db_status,
            track_feedback as _track_feedback,   # Gauge database_status
        #    request_counter as _request_counter
        )
        update_db_status = _update_db_status
        track_feedback = _track_feedback
        # request_counter = _request_counter
        print("✅ Prometheus tracking functions loaded")
    except ImportError as e:
        ENABLE_PROMETHEUS = False  # Désactivation silencieuse
        print(f"⚠️  Prometheus tracking not available: {e}")
        

if ENABLE_DISCORD:
    try:
        from src.monitoring.discord_notifier import (
            alert_high_latency as _alert_high_latency,
            alert_database_disconnected as _alert_database_disconnected,
            notifier as _notifier  # Instance DiscordNotifier globale
        )
        alert_high_latency = _alert_high_latency
        alert_database_disconnected = _alert_database_disconnected
        notifier = _notifier
        print("✅ Discord notifier loaded")
    except ImportError as e:
        ENABLE_DISCORD = False
        print(f"⚠️  Discord notifier not available: {e}")

TEMPLATES_DIR = ROOT_DIR / "src" / "web" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()

predictor = CatDogPredictor()


@router.get("/", response_class=HTMLResponse, tags=["🌐 Page Web"])
async def welcome(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,  # Requis par Jinja2
        "model_loaded": predictor.is_loaded()  # Affiche warning si modèle absent
    })

@router.get("/info", response_class=HTMLResponse, tags=["🌐 Page Web"])
async def info_page(request: Request):
    model_info = {
        "name": "Cats vs Dogs Classifier",
        "version": "3.0.0",  # 🆕 V3
        "description": "Modèle CNN pour classification chats/chiens",
        "parameters": predictor.model.count_params() if predictor.is_loaded() else 0,
        # 📊 Nombre de paramètres (ex: ~23M pour VGG16 fine-tuned)
        "classes": ["Cat", "Dog"],
        "input_size": f"{predictor.image_size[0]}x{predictor.image_size[1]}",
        # 🖼️ Dimension attendue (ex: 224x224)
        "model_loaded": predictor.is_loaded(),
        # 🆕 V3 - Informations monitoring
        "prometheus_enabled": ENABLE_PROMETHEUS,
        "discord_enabled": ENABLE_DISCORD
    }
    return templates.TemplateResponse("info.html", {
        "request": request, 
        "model_info": model_info
    })

@router.get("/inference", response_class=HTMLResponse, tags=["🧠 Inférence"])
async def inference_page(request: Request):
    return templates.TemplateResponse("inference.html", {
        "request": request,
        "model_loaded": predictor.is_loaded()
    })

@router.post("/api/predict", tags=["🧠 Inférence"])
async def predict_api(
    file: UploadFile = File(...),
    rgpd_consent: bool = Form(False),
    token: str = Depends(verify_token),  # 🔐 Authentification requise
    db: Session = Depends(get_db)       # 🗄️ Injection session DB
):
    if not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Format d'image invalide")
        
    
    start_time = time.perf_counter()
   
    try:
        image_data = await file.read()
        
        result = predictor.predict(image_data)
        end_time = time.perf_counter()
        inference_time_ms = int((end_time - start_time) * 1000)
        proba_cat = result['probabilities']['cat'] * 100  
        proba_dog = result['probabilities']['dog'] * 100
        feedback_record = FeedbackService.save_prediction_feedback(
            db=db,
            inference_time_ms=inference_time_ms,
            success=True,
            prediction_result=result["prediction"].lower(),  # 'cat' ou 'dog'
            proba_cat=proba_cat,
            proba_dog=proba_dog,
            rgpd_consent=rgpd_consent,
            filename=file.filename if rgpd_consent else None,  # Anonymisation
            user_feedback=None,  # Sera mis à jour via /api/update-feedback
            user_comment=None
        )
        response_data = {
            "filename": file.filename,
            "prediction": result["prediction"],  # "Cat" ou "Dog"
            "confidence": f"{result['confidence']:.2%}",  # "95.34%"
            "probabilities": {
                "cat": f"{result['probabilities']['cat']:.2%}",
                "dog": f"{result['probabilities']['dog']:.2%}"
            },
            "inference_time_ms": inference_time_ms,
            "feedback_id": feedback_record.id  # Pour update feedback ultérieur
        }
        inference_time_ms = (time.time() - start_time) * 1000
        track_inference_time(inference_time_ms)
        return response_data
        
    except Exception as e:
        end_time = time.perf_counter()
        inference_time_ms = int((end_time - start_time) * 1000)
        
        try:
            FeedbackService.save_prediction_feedback(
                db=db,
                inference_time_ms=inference_time_ms,
                success=False,  # Marqueur échec
                prediction_result="error",
                proba_cat=0.0,
                proba_dog=0.0,
                rgpd_consent=False,
                filename=None,
                user_feedback=None,
                user_comment=str(e)  
            )
        except:
            pass 
        
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@router.post("/api/update-feedback", tags=["📊 Monitoring"])
async def update_feedback(
    feedback_id: int = Form(...),        
    user_feedback: int = Form(None),     
    user_comment: str = Form(None),      
    db: Session = Depends(get_db)
):
    
    try:
        from src.database.models import PredictionFeedback
        record = db.query(PredictionFeedback).filter(
            PredictionFeedback.id == feedback_id
        ).first()
        
        if not record:
            raise HTTPException(
                status_code=404,
                detail="Enregistrement de feedback non trouvé"
            )
        if not record.rgpd_consent:
            raise HTTPException(
                status_code=403,
                detail="Consentement RGPD non accepté. Impossible de stocker le feedback."
            )
        if user_feedback is not None:
            if user_feedback not in [0, 1]:
                raise HTTPException(
                    status_code=400,
                    detail="user_feedback doit être 0 ou 1"
                )
            record.user_feedback = user_feedback
            
            if ENABLE_PROMETHEUS:
                try:
                    track_feedback("positive" if user_feedback == 1 else "negative")
                except Exception as e:
                    print(f"Soucis avec le feedback Prometheus: {e}")
        
        if user_comment:
            record.user_comment = user_comment
        
        db.commit()
        
    except HTTPException:
        raise  
    except Exception as e:
        db.rollback()  
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la mise à jour: {str(e)}"
        )

@router.get("/api/statistics", tags=["📊 Monitoring"])
async def get_statistics(db: Session = Depends(get_db)):
    try:
        stats = FeedbackService.get_statistics(db)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des statistiques: {str(e)}"
        )

@router.get("/api/recent-predictions", tags=["📊 Monitoring"])
async def get_recent_predictions(
    limit: int = 10,  # Nombre de résultats (défaut : 10)
    db: Session = Depends(get_db)
):
    try:
        predictions = FeedbackService.get_recent_predictions(db, limit=limit)
        results = []
        for pred in predictions:
            results.append({
                "id": pred.id,
                "timestamp": pred.timestamp.isoformat() if pred.timestamp else None,
                # ISO 8601 : "2025-11-16T14:32:00.123456"
                "prediction_result": pred.prediction_result,
                "proba_cat": float(pred.proba_cat),  # Decimal → float
                "proba_dog": float(pred.proba_dog),
                "inference_time_ms": pred.inference_time_ms,
                "success": pred.success,
                "rgpd_consent": pred.rgpd_consent,
                "user_feedback": pred.user_feedback,
                "filename": pred.filename if pred.rgpd_consent else None
                # 🔐 Anonymisation : filename uniquement si consent
            })
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des prédictions: {str(e)}"
        )

@router.get("/api/info", tags=["🧠 Inférence"])
async def api_info():
    return {
        "model_loaded": predictor.is_loaded(),
        "model_path": str(predictor.model_path),
        "version": "3.0.0",  # 🆕 V3
        "parameters": predictor.model.count_params() if predictor.is_loaded() else 0,
        "features": [
            "Image classification (cats/dogs)",
            "RGPD compliance",
            "User feedback collection",
            "PostgreSQL monitoring",
            "Prometheus metrics" if ENABLE_PROMETHEUS else None,  # 🆕 V3
            "Discord alerting" if ENABLE_DISCORD else None  # 🆕 V3
        ],
        "monitoring": {  # 🆕 V3 - Détails monitoring externe
            "prometheus_enabled": ENABLE_PROMETHEUS,
            "discord_enabled": ENABLE_DISCORD,
            "metrics_endpoint": "/metrics" if ENABLE_PROMETHEUS else None
        }
    }

@router.get("/monitoring", response_class=HTMLResponse, tags=["📊 Monitoring"])
async def monitoring_dashboard(request: Request, db: Session = Depends(get_db)):
    try:
        dashboard_data = DashboardService.get_dashboard_data(db)
        dashboard_data["grafana_url"] = "http://localhost:3000" if ENABLE_PROMETHEUS else None
        dashboard_data["prometheus_url"] = "http://localhost:9090" if ENABLE_PROMETHEUS else None
        
        return templates.TemplateResponse("monitoring.html", {
            "request": request,
            **dashboard_data  
        })
    except Exception as e:
        return templates.TemplateResponse("monitoring.html", {
            "request": request,
            "error": f"Erreur lors du chargement des données : {str(e)}"
        })

@router.get("/health", tags=["💚 Santé système"])
async def health_check(db: Session = Depends(get_db)):
    db_status = "connected"
    db_connected = True
    
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        
    except Exception as e:
        db_status = f"error: {str(e)}"
        db_connected = False
        if ENABLE_DISCORD:
            try:
                if alert_database_disconnected:
                    alert_database_disconnected()
            except Exception as discord_error:
                print(f"⚠️  Discord alert failed: {discord_error}")
                
    if ENABLE_PROMETHEUS and update_db_status:
        try:
            update_db_status(db_connected)
        except Exception as e:
            print(f"⚠️  Prometheus status update failed: {e}")

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "model_loaded": predictor.is_loaded(),
        "database": db_status,
        "monitoring": {
            "prometheus": ENABLE_PROMETHEUS,
            "discord": ENABLE_DISCORD
        }
    }
    
