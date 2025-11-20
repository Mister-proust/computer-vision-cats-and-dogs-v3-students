"""
═══════════════════════════════════════════════════════════════════════════════
📢 DISCORD NOTIFIER - Système d'alertes temps réel
═══════════════════════════════════════════════════════════════════════════════

🎯 OBJECTIF PÉDAGOGIQUE
Module d'alerting via Discord Webhooks, complétant le monitoring Grafana.
Illustre l'importance d'avoir plusieurs canaux de notification pour les incidents critiques.

📚 CONCEPTS CLÉS
- Webhooks : callbacks HTTP pour intégrations tierces
- Alerting multi-canal : Discord (instant) + Grafana (dashboards)
- Fail-safe : notifications même si Grafana down
- Severity levels : classification des alertes (info, warning, error, critical)

🔗 INTÉGRATION
- Appelé par : src/api/main.py (incidents métier), healthchecks, deploy.yml
- Complémentaire à : Grafana Unified Alerting (plus avancé mais nécessite infra)

⚠️ STATUT : Provisoire (fallback si Grafana non configuré)
En production : privilégier Grafana alerting + notification Discord via contact point

═══════════════════════════════════════════════════════════════════════════════
"""
import os
import requests
from datetime import datetime
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 📂 CHARGEMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent
# 📁 Remonte de 3 niveaux : monitoring/ → src/ → racine projet
# Exemple : src/monitoring/discord_notifier.py → computer-vision-cats-and-dogs-v3/

load_dotenv(ROOT_DIR / '.env')
# 🔐 Charge variables depuis .env (DISCORD_WEBHOOK_URL)
# Utilise python-dotenv (déjà dans requirements/monitoring.txt)

# ═══════════════════════════════════════════════════════════════════════════
# 📢 CLASSE PRINCIPALE - Notifier Discord
# ═══════════════════════════════════════════════════════════════════════════
class DiscordNotifier:
    """
    Envoie des notifications Discord pour événements critiques
    
    💡 DISCORD WEBHOOKS
    Les webhooks Discord permettent d'envoyer des messages sans bot complexe :
    - URL unique par canal (format : https://discord.com/api/webhooks/{id}/{token})
    - POST JSON → message apparaît instantanément
    - Embeds : messages enrichis (couleurs, champs, timestamps)
    
    🔧 SETUP WEBHOOK
    1. Discord → Paramètres du serveur → Intégrations → Webhooks
    2. Créer Webhook → Copier URL
    3. Ajouter dans .env : DISCORD_WEBHOOK_URL=https://...
    
    📊 ALTERNATIVES
    - Slack : webhooks similaires (format légèrement différent)
    - Telegram : bot API (plus complexe mais notifications push mobile)
    - Email : SMTP (moins temps réel, risque spam)
    """
    
    def __init__(self):
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        # 🔗 Récupère URL depuis variables d'environnement
        # Format attendu : https://discord.com/api/webhooks/{id}/{token}
        
        self.enabled = bool(self.webhook_url)
        # ✅ Active uniquement si webhook configuré (graceful degradation)
        # Si absent → send_alert() sera no-op (pas d'erreur)
        
    def send_alert(self, 
                   title: str, 
                   message: str, 
                   level: str = "info",
                   metrics: Optional[dict] = None):
        """
        Envoie une alerte Discord enrichie (embed)
        
        Args:
            title: Titre de l'alerte (ex: "Model Performance Degradation")
            message: Description détaillée du problème
            level: Sévérité (info/warning/error/critical) → détermine la couleur
            metrics: Dict optionnel de métriques {nom: valeur} affichées en champs
        
        💡 EMBEDS DISCORD
        Format riche avec :
        - Couleur (barre latérale) : code visuel de sévérité
        - Champs : key-value pairs (ex: Accuracy: 78%, Threshold: 85%)
        - Timestamp : horodatage UTC automatique
        - Footer : signature du bot
        
        📊 EXEMPLE DE RENDU
        ┌─ 🚨 Model Performance Degradation ────────────────┐ (barre rouge)
        │ Model accuracy (78%) dropped below threshold (85%)│
        │                                                    │
        │ Current Accuracy: 78%     Threshold: 85%          │
        │                                                    │
        │ CV Cats & Dogs Monitoring • 2025-11-16 14:32 UTC  │
        └────────────────────────────────────────────────────┘
        """
        if not self.enabled:
            return  # Sortie silencieuse si webhook non configuré
            
        # ─────────────────────────────────────────────────────────────────────
        # 🎨 MAPPING COULEURS (format Discord : entier décimal)
        # ─────────────────────────────────────────────────────────────────────
        colors = {
            "info": 3447003,      # Bleu (#3498db) - informations générales
            "warning": 16776960,  # Jaune (#ffff00) - attention requise
            "error": 15158332,    # Rouge (#e74c3c) - dysfonctionnement
            "critical": 10038562  # Rouge foncé (#992d22) - incident majeur
        }
        # 💡 Conversion hex → décimal : int("3498db", 16) = 3447003
        # Visuel : couleur de la barre latérale de l'embed
        
        # ─────────────────────────────────────────────────────────────────────
        # 📦 CONSTRUCTION EMBED (format Discord API)
        # ─────────────────────────────────────────────────────────────────────
        embed = {
            "title": f"🚨 {title}",
            # 🏷️ Titre avec emoji pour attention visuelle
            # Limite Discord : 256 caractères
            
            "description": message,
            # 📝 Corps du message (détails du problème)
            # Limite Discord : 4096 caractères
            
            "color": colors.get(level, 3447003),
            # 🎨 Couleur de la barre latérale (défaut : bleu info)
            
            "timestamp": datetime.utcnow().isoformat(),
            # ⏰ Horodatage ISO 8601 (ex: 2025-11-16T14:32:00.123456)
            # Discord convertit automatiquement en heure locale de l'utilisateur
            
            "footer": {
                "text": "CV Cats & Dogs Monitoring"
            }
            # 📌 Signature en bas de l'embed (branding)
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # 📊 AJOUT MÉTRIQUES (si fournies)
        # ─────────────────────────────────────────────────────────────────────
        if metrics:
            embed["fields"] = [
                {
                    "name": key,           # Nom de la métrique
                    "value": str(value),   # Valeur (converti en string)
                    "inline": True         # Affichage côte à côte (max 3 par ligne)
                }
                for key, value in metrics.items()
            ]
            # 💡 EXEMPLE RENDU
            # metrics = {"Accuracy": "78%", "Threshold": "85%", "Gap": "-7%"}
            # → 3 champs inline affichés horizontalement
            # Limite Discord : 25 champs max par embed
        
        # ─────────────────────────────────────────────────────────────────────
        # 🚀 PAYLOAD COMPLET (webhook Discord)
        # ─────────────────────────────────────────────────────────────────────
        payload = {
            "username": "MLOps Bot",
            # 🤖 Nom affiché pour le bot (override webhook par défaut)
            # Optionnel : "avatar_url" pour icône custom
            
            "embeds": [embed]
            # 📋 Liste d'embeds (Discord supporte jusqu'à 10 par message)
            # Ici : 1 seul embed par alerte (clarté)
        }
        
        # ─────────────────────────────────────────────────────────────────────
        # 📡 ENVOI HTTP POST
        # ─────────────────────────────────────────────────────────────────────
        try:
            response = requests.post(self.webhook_url, json=payload)
            # 🌐 POST vers Discord API
            # json=payload : sérialise auto en JSON + header Content-Type
            
            response.raise_for_status()
            # ✅ Lève exception si status ≠ 2xx (ex: 400 Bad Request, 404 Not Found)
            # Codes Discord courants :
            #   - 204 No Content : succès
            #   - 400 : payload invalide (embed trop long, champ manquant)
            #   - 404 : webhook supprimé ou URL invalide
            #   - 429 : rate limit (5 messages/2s par webhook)
            
        except Exception as e:
            # 🛡️ FAIL-SAFE : erreur Discord ne doit PAS crasher l'app
            print(f"❌ Failed to send Discord alert: {e}")
            # En production : logger au lieu de print
            # Alternative : retry logic avec backoff exponentiel

# ═══════════════════════════════════════════════════════════════════════════
# 🌍 INSTANCE GLOBALE (singleton pattern)
# ═══════════════════════════════════════════════════════════════════════════
notifier = DiscordNotifier()
# 💡 Instanciation unique au module load
# Avantage : webhook_url chargé 1 seule fois (performance)
# Usage : from discord_notifier import notifier; notifier.send_alert(...)

# ═══════════════════════════════════════════════════════════════════════════
# 🛠️ FONCTIONS HELPER - Alertes prédéfinies
# ═══════════════════════════════════════════════════════════════════════════
# Simplifient l'usage depuis l'API (interface de haut niveau)

def alert_model_degradation(accuracy: float, threshold: float = 0.85):
    """
    Alerte si l'accuracy du modèle baisse sous le seuil
    
    🎯 CAS D'USAGE
    - Data drift : distribution des inputs change
    - Concept drift : relation inputs-outputs évolue
    - Model staleness : modèle non ré-entraîné depuis longtemps
    
    🔗 APPELÉ DEPUIS
    - Endpoint /feedback (calcul accuracy glissante sur derniers N feedbacks)
    - Script de monit oring périodique (cron job)
    
    Args:
        accuracy: Accuracy actuelle (ex: 0.78 pour 78%)
        threshold: Seuil minimal acceptable (défaut: 85%)
    
    💡 EXEMPLE INTÉGRATION API
    recent_feedbacks = db.get_last_n_feedbacks(100)
    accuracy = sum(f.correct for f in recent_feedbacks) / len(recent_feedbacks)
    alert_model_degradation(accuracy)
    """
    if accuracy < threshold:
        notifier.send_alert(
            title="Model Performance Degradation",
            message=f"Model accuracy ({accuracy:.2%}) dropped below threshold ({threshold:.2%})",
            level="warning",  # Warning car dégradation progressive (pas incident immédiat)
            metrics={
                "Current Accuracy": f"{accuracy:.2%}",
                "Threshold": f"{threshold:.2%}",
                "Gap": f"{(accuracy - threshold):.2%}"  # Négatif = problème
            }
        )

def alert_high_latency(latency_ms: float, threshold: float = 2000):
    """
    Alerte si la latence d'inférence est trop élevée
    
    🎯 CAS D'USAGE
    - Surcharge serveur (trop de requêtes simultanées)
    - GPU saturé ou non disponible
    - Problème réseau (si modèle distant)
    
    🔗 APPELÉ DEPUIS
    - Endpoint POST /predict (après mesure temps inférence)
    - Middleware FastAPI (tracking latence globale)
    
    Args:
        latency_ms: Temps d'inférence en millisecondes
        threshold: Seuil maximal tolérable (défaut: 2000ms = 2s)
    
    💡 SEUILS TYPIQUES
    - <100ms : excellent (temps réel)
    - 100-500ms : acceptable (perception fluide)
    - 500-2000ms : dégradé (utilisateur impatient)
    - >2000ms : critique (timeout probable)
    """
    if latency_ms > threshold:
        notifier.send_alert(
            title="High Inference Latency",
            message=f"Inference taking {latency_ms}ms (threshold: {threshold}ms)",
            level="error",  # Error car impact direct sur UX
            metrics={
                "Latency": f"{latency_ms:.0f}ms",
                "Threshold": f"{threshold:.0f}ms",
                "Slowdown": f"x{(latency_ms / threshold):.1f}"  # Ex: x2.5 = 2.5x plus lent
            }
        )

def alert_database_disconnected():
    """
    Alerte si la base de données PostgreSQL est déconnectée
    
    🎯 CAS D'USAGE
    - Crash PostgreSQL (OOM, corruption)
    - Problème réseau (firewall, DNS)
    - Credentials invalides (rotation password)
    
    🔗 APPELÉ DEPUIS
    - Healthcheck API (/health endpoint)
    - Exception handlers SQLAlchemy (database errors)
    - Prometheus metric update (database_status.set(0))
    
    💡 WORKFLOW TYPIQUE
    1. Exception SQLAlchemy levée (connection timeout)
    2. API appelle alert_database_disconnected()
    3. Discord notifie équipe DevOps
    4. Grafana dashboard montre aussi db_status=0 (confirmation visuelle)
    5. Équipe diagnostique (logs PostgreSQL, docker ps, etc.)
    """
    notifier.send_alert(
        title="Database Connection Lost",
        message="PostgreSQL database is unreachable. All feedback storage is currently disabled.",
        level="critical",  # Critical car perte de fonctionnalité majeure
        metrics={
            "Service": "PostgreSQL",
            "Impact": "❌ Feedback storage offline",
            "Action": "Check docker logs cv_postgres"
        }
    )

def alert_deployment_success(version: str):
    """
    Notification de déploiement réussi (non-blocking, informatif)
    
    🎯 CAS D'USAGE
    - Confirmation déploiement CI/CD (GitHub Actions)
    - Traçabilité des releases (audit trail)
    - Synchronisation équipe (awareness)
    
    🔗 APPELÉ DEPUIS
    - GitHub Actions (deploy.yml, dernière étape)
    - Script de déploiement manuel
    
    Args:
        version: Identifiant de version (ex: "v3.2.1" ou git commit hash)
    
    💡 COMPLÉMENT GRAFANA ANNOTATION
    En production, aussi créer annotation Grafana :
    - Marque verticale sur dashboards (timeline)
    - Corrélation : déploiement ↔ changements métriques
    """
    notifier.send_alert(
        title="Deployment Successful",
        message=f"Version {version} deployed successfully to production",
        level="info",  # Info car événement positif (pas un problème)
        metrics={
            "Version": version,
            "Status": "✅ Running",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

# ═══════════════════════════════════════════════════════════════════════════
# 🎓 ÉVOLUTIONS POSSIBLES (pour aller plus loin)
# ═══════════════════════════════════════════════════════════════════════════
#
# 1. RATE LIMITING LOCAL
#    Éviter spam si alerte déclenchée en boucle :
#    from functools import lru_cache
#    from time import time
#    
#    last_alert = {}
#    def alert_with_cooldown(alert_type, cooldown_seconds=300):
#        now = time()
#        if now - last_alert.get(alert_type, 0) > cooldown_seconds:
#            # Envoyer alerte
#            last_alert[alert_type] = now
#
# 2. ALERTING MULTI-CANAL
#    class MultiChannelNotifier:
#        def __init__(self):
#            self.discord = DiscordNotifier()
#            self.slack = SlackNotifier()
#            self.email = EmailNotifier()
#        
#        def send_critical(self, ...):
#            # Critical → tous canaux
#            self.discord.send_alert(...)
#            self.slack.send_alert(...)
#            self.email.send_alert(...)
#
# 3. TEMPLATES D'ALERTES
#    from jinja2 import Template
#    TEMPLATES = {
#        "model_degradation": Template("Accuracy {{ acc }} < {{ threshold }}")
#    }
#    → Uniformise les messages, facilite i18n
#
# 4. WEBHOOK SIGNATURE VERIFICATION
#    Pour sécuriser webhook entrant (si bidirectionnel) :
#    - Discord signe les requêtes (header X-Signature-Ed25519)
#    - Vérifier signature avant traiter commande bot
#
# 5. RICH EMBEDS (images, graphiques)
#    embed["image"] = {"url": "https://quickchart.io/chart?c={...}"}
#    → Afficher graphique accuracy directement dans Discord
#
# ═══════════════════════════════════════════════════════════════════════════
# 📚 RESSOURCES
# ═══════════════════════════════════════════════════════════════════════════
#
# - Discord Webhook API: https://discord.com/developers/docs/resources/webhook
# - Embed visualizer: https://leovoel.github.io/embed-visualizer/
# - Rate limits: https://discord.com/developers/docs/topics/rate-limits
# - Alerting best practices: https://landing.google.com/sre/sre-book/chapters/practical-alerting/
#
# ═══════════════════════════════════════════════════════════════════════════