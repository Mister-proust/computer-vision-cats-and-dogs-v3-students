# Migration V2 → V3

## Compatibilité
✅ **100% rétrocompatible** - Votre code V2 continue de fonctionner

## Étapes de Migration

### 1. Ajout des Fichiers (5 min)
```bash
# Créer les nouveaux dossiers
mkdir -p docker monitoring/{prometheus,grafana,alertmanager}
f
# Copier les fichiers depuis ce guide
# - docker/docker-compose.yml
# - docker/Dockerfile.app
# - monitoring/prometheus/prometheus.yml
# - etc.
```

### 2. Installation Dépendances (2 min)
```bash
pip install -r requirements/monitoring.txt
```

### 3. Configuration (5 min)
```bash
# Copier et adapter .env
cp .env.example .env
# Éditer .env avec vos valeurs
```

### 4. Test Local (10 min)
```bash
# Démarrer la stack Docker
make docker-up

# Vérifier les services
docker ps
curl http://localhost:8007/health
curl http://localhost:3007  # Grafana
curl http://localhost:9097  # Prometheus
```

### 5. Migration Progressive
- **Phase 1** : Mode hybride (Plotly V2 + Grafana V3 en parallèle)
- **Phase 2** : Tests Grafana + Discord
- **Phase 3** : Production complète

## Rollback
Si besoin de revenir en V2 :
```bash
# Arrêter Docker
make docker-down

# Relancer en mode V2 classique
python scripts/run_api.py
```