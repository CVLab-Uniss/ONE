import os
import json
import time
import threading
import math
import re
import logging
from typing import Any, Dict, Optional, Tuple, List

import base64
import hmac
import hashlib
from urllib.parse import quote, urlparse

import functools

from functools import wraps
from flask import Flask, request, jsonify, current_app, has_app_context
import requests
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

from kubernetes import client, config
from kubernetes.client.rest import ApiException

import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from email.mime.text import MIMEText

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "MySecureToken1234")
IOTHUB_CONNECTION_STRING = os.getenv("IOTHUB_CONNECTION_STRING", "")
K8S_TARGET_NAMESPACE = os.getenv("K8S_TARGET_NAMESPACE", "dev")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo-service:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "one_project_db")
BASE_DOMAIN = os.getenv("BASE_DOMAIN", "128.203.65.69.nip.io")

# immagini default
# Nome e immagine del workload "di base" che gira sempre sull'edge
DEFAULT_WORKLOAD_NAME  = os.environ.get("DEFAULT_WORKLOAD_NAME", "myWorkloadModule")
DEFAULT_WORKLOAD_IMAGE = os.environ.get(
    "DEFAULT_WORKLOAD_IMAGE",
    "pietroruiu/one-project:wlchar_ad_api_v3_cpu",
)

# Nome del modulo di job di ReID
DETECTION_JOB_MODULE = os.environ.get("DETECTION_JOB_MODULE", "myDetectionJob")


# slave
SLAVE_IMAGE = os.getenv("SLAVE_IMAGE", "mfadd/slave-node:v16-iotedge-gateway")
SLAVE_SHARED_TOKEN = os.getenv("SLAVE_SHARED_TOKEN", "edge-temporary-token")
SLAVE_CONTAINER_PORT = int(os.getenv("SLAVE_CONTAINER_PORT", "8080"))
HEALTHCHECK_TIMEOUT = int(os.getenv("HEALTHCHECK_TIMEOUT", "25"))

# job di detection (ReID)
DETECTION_JOB_IMAGE = os.getenv("DETECTION_JOB_IMAGE", "pietroruiu/one-project:wlchar_reid_api_v3_cpu")

# master URL che il job deve contattare
MASTER_BASE_HOST = os.getenv("MASTER_BASE_HOST", "128.203.65.69.nip.io")
MASTER_SCAN_REPORT_PATH = "/api/v1/scan/report"
MASTER_REPORT_URL = f"{MASTER_BASE_HOST}{MASTER_SCAN_REPORT_PATH}"

SCAN_DEFAULT_TIMEOUT_S = 300
SCAN_HARD_TIMEOUT_S = int(os.getenv("SCAN_HARD_TIMEOUT_S", "400"))

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "master@one-project.local")

# sessioni attive
SCAN_SESSIONS: dict = {}

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [master-node] %(message)s",
)
log = logging.getLogger("master-node")

# ------------------------------------------------------------------------------
# Mongo (robusto, un solo client)
# ------------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo-service:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "one_project_db")

_mongo = MongoClient(MONGO_URI)
db = _mongo[MONGO_DB_NAME]

scan_sessions = db["scan_sessions"]
scan_events   = db["scan_events"]

# Indici idempotenti
scan_sessions.create_index([("session_id", ASCENDING)], unique=True)
scan_sessions.create_index("expires_at", expireAfterSeconds=0)  # TTL su expires_at

scan_events.create_index([("session_id", ASCENDING)])
scan_events.create_index([("created_at", ASCENDING)])

# ------------------------------------------------------------------------------
# Flask
# ------------------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------------------
# Auth decorators
# ------------------------------------------------------------------------------
def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        # formato atteso: "Bearer <token>"
        if auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1].strip()
        else:
            token = auth.strip()

        # accetta sia l'admin che il token condiviso per i job edge
        allowed_tokens = {ADMIN_TOKEN, SLAVE_SHARED_TOKEN}
        if token not in allowed_tokens:
            log.warning("unauthorized request: got token=%r", token)
            return jsonify({"ok": False, "error": "unauthorized"}), 401

        return fn(*args, **kwargs)
    return wrapper



def require_bearer(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        hdr = request.headers.get("Authorization", "")
        # accetta SIA il token admin SIA il token edge
        if hdr not in (f"Bearer {ADMIN_TOKEN}", f"Bearer {SLAVE_SHARED_TOKEN}"):
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper

def _get_bearer_token(req):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return auth.split(" ", 1)[1].strip()

def require_edge_auth(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = _get_bearer_token(request)
        edge_token = os.getenv("SLAVE_SHARED_TOKEN", "")
        admin_token = os.getenv("ADMIN_TOKEN", "")

        # Se il token è corretto → ok
        if token and (token == edge_token or token == admin_token):
            return fn(*args, **kwargs)

        # Se è mancante o "None" → accettiamo SENZA warning
        if token is None or token == "None":
            return fn(*args, **kwargs)

        # Altro valore strano → log a livello WARNING ma accettiamo
        current_app.logger.warning(
            "[scan-report] unexpected Authorization token=%r – accepting anyway for ReID jobs",
            token,
        )
        return fn(*args, **kwargs)

    return wrapper




def auth_required(fn):
    # alias se ti serve un altro nome
    return require_bearer(fn)

def pick_zone_for_coords(lat: float, lon: float) -> str:
    """
    Determina automaticamente la zona (provincia sarda) a partire dalle coordinate.
    1. prova a capire in che provincia cade (bounding box)
    2. se esiste già almeno uno slave per quella zona, la usa
    3. altrimenti prova a crearla
    """

    # Bounding box molto ampi per la Sardegna, così evitiamo buchi.
    # Se vuoi affinarli li stringi più avanti.
    PROVINCE_BOUNDS = {
        # centro-ovest
        "oristano": {
            "lat_min": 39.6,
            "lat_max": 40.3,
            "lon_min": 8.3,
            "lon_max": 9.0,
        },
        # sud area cagliari
        "cagliari": {
            "lat_min": 38.8,
            "lat_max": 39.5,
            "lon_min": 8.9,
            "lon_max": 9.6,
        },
        # provincia “nuova” sud sardegna: prendiamo fascia più larga nel sud-ovest
        "sud-sardegna": {
            "lat_min": 38.6,
            "lat_max": 39.5,
            "lon_min": 8.2,
            "lon_max": 9.1,
        },
        # nord-ovest
        "sassari": {
            "lat_min": 40.5,
            "lat_max": 41.3,
            "lon_min": 8.1,
            "lon_max": 9.2,
        },
        # centro-nord interno
        "nuoro": {
            "lat_min": 40.0,
            "lat_max": 40.6,
            "lon_min": 9.0,
            "lon_max": 9.7,
        },
        # nord-est / costa smeralda
        "olbia-tempio": {
            "lat_min": 40.7,
            "lat_max": 41.3,
            "lon_min": 9.2,
            "lon_max": 10.1,
        },
        # fascia est centrale
        "ogliastra": {
            "lat_min": 39.6,
            "lat_max": 40.3,
            "lon_min": 9.3,
            "lon_max": 10.1,
        },
        # sud-ovest storico
        "carbonia-iglesias": {
            "lat_min": 38.8,
            "lat_max": 39.5,
            "lon_min": 8.1,
            "lon_max": 8.9,
        },
    }

    guessed_zone = "default"
    for prov, box in PROVINCE_BOUNDS.items():
        if (
            box["lat_min"] <= lat <= box["lat_max"]
            and box["lon_min"] <= lon <= box["lon_max"]
        ):
            guessed_zone = prov
            break

    app.logger.info(f"[zone-picker] coordinate ({lat}, {lon}) → zona '{guessed_zone}'")

    # 2) controllo se nel cluster esiste già uno slave con quella zona
    try:
        gateways = list_all_slave_gateways(K8S_TARGET_NAMESPACE)
        for gw in gateways:
            if gw.get("zone") == guessed_zone:
                app.logger.info(
                    f"[zone-picker] uso slave esistente per zona '{guessed_zone}'"
                )
                return guessed_zone
    except Exception as e:
        app.logger.warning(
            f"[zone-picker] impossibile leggere gli slave esistenti: {e}"
        )

    # 3) se non esiste, lo creiamo adesso
    try:
        ensure_slave_gateway(guessed_zone)
        app.logger.info(f"[zone-picker] creato slave per zona '{guessed_zone}'")
    except Exception as e:
        app.logger.error(
            f"[zone-picker] errore creando lo slave per '{guessed_zone}': {e}"
        )

    return guessed_zone

def save_scan_session(session_id: str, data: dict, ttl_seconds: int):
    """
    Salva la sessione su Mongo.
    - created_at viene fissato alla prima insert e NON viene più toccato
    - expires_at è mobile (TTL per la collection)
    """
    now = datetime.utcnow()
    expires_at = now + timedelta(seconds=int(max(ttl_seconds, 60)))

    scan_sessions.update_one(
        {"session_id": session_id},
        {
            # aggiorniamo dati e TTL ad ogni chiamata
            "$set": {
                "data": data,
                "expires_at": expires_at,
            },
            # created_at viene scritto SOLO alla prima insert
            "$setOnInsert": {
                "created_at": now,
            },
        },
        upsert=True,
    )


def get_scan_session(session_id: str) -> Optional[dict]:
    doc = scan_sessions.find_one({"session_id": session_id})
    return (doc or {}).get("data") if doc else None

def delete_scan_session(session_id: str):
    scan_sessions.delete_one({"session_id": session_id})

def store_scan_event(session_id: str, device_id: str, zone: Optional[str], event: str, details: dict, timestamp: Optional[str]):
    scan_events.insert_one({
        "session_id": session_id,
        "device_id": device_id,
        "zone": zone,
        "event": event,
        "details": details or {},
        "timestamp": timestamp,
        "created_at": datetime.utcnow(),
    })


def register_scan_session_in_memory(session_id: str, session_data: dict, devices_info: dict, slave_url: str):
    """
    Registra la sessione anche in memoria per il GC 'mobile':
      - expires_at: timeout_s (es. 180s) SLIDING, resettato ad ogni evento
      - hard_expires_at: hard_timeout_s (es. 400s) FISSO
      - devices: mappa device_id -> {lat, lon, slave_url, last_event_at}
    """
    now = time.time()
    timeout_s = int(session_data.get("timeout_s", SCAN_DEFAULT_TIMEOUT_S))
    hard_timeout_s = int(session_data.get("hard_timeout_s", SCAN_HARD_TIMEOUT_S))

    devices = {}
    for dev_id, info in (devices_info or {}).items():
        devices[dev_id] = {
            "lat": info.get("lat"),
            "lon": info.get("lon"),
            "slave_url": info.get("slave_url", slave_url),
            "last_event_at": now,
        }

    SCAN_SESSIONS[session_id] = {
        "timeout_s": timeout_s,
        "hard_timeout_s": hard_timeout_s,
        "created_at": now,
        "expires_at": now + timeout_s,          # timeout "mobile"
        "hard_expires_at": now + hard_timeout_s, # timeout "duro"
        "center": dict(session_data.get("center") or {}),
        "radius_m": float(session_data.get("radius_m", 300.0)),
        "zone": session_data.get("zone"),
        "slave_url": slave_url,
        "job_image": session_data.get("job_image") or DETECTION_JOB_IMAGE,
        "devices": devices,
    }

    log.info(
        "[scan] registered in-memory session %s (timeout=%ss, hard=%ss, radius=%sm, devices=%s)",
        session_id, timeout_s, hard_timeout_s,
        session_data.get("radius_m", 300),
        ", ".join(devices.keys()) or "none",
    )


def recompute_active_devices_for_session(session_id: str):
    """
    Ricalcola quali device devono essere attivi nella sessione:
      - chiede allo slave la lista di edge entro radius_m dal center
      - per i device che NON sono più nella lista → revert a myWorkloadModule
      - per i nuovi device nella lista → push del myDetectionJob
    """
    sess = SCAN_SESSIONS.get(session_id)
    if not sess:
        return

    center = sess.get("center") or {}
    c_lat = center.get("lat")
    c_lon = center.get("lon")
    radius_m = float(sess.get("radius_m", 300.0))
    zone = sess.get("zone")
    slave_url = sess.get("slave_url")

    if c_lat is None or c_lon is None:
        log.warning("[recompute] session %s center unknown, skip recompute", session_id)
        return

    # Se manca lo slave_url, proviamo a recuperarlo dalla zone
    if not slave_url and zone:
        try:
            slave_info = ensure_slave_gateway(zone)
            slave_url = slave_info.get("url", "")
            sess["slave_url"] = slave_url
        except Exception:
            log.exception("[recompute] ensure_slave_gateway failed for zone %s", zone)
            return

    if not slave_url:
        log.warning("[recompute] session %s has no slave_url, skip", session_id)
        return

    # 1) chiediamo allo slave i device entro il raggio dal nuovo centro
    edges_resp = query_slave_edges(slave_url, float(c_lat), float(c_lon), radius_m)
    if not edges_resp.get("ok"):
        log.warning("[recompute] query_slave_edges failed for session %s: %s", session_id, edges_resp)
        return

    edges = edges_resp.get("edges", []) or []

        # Se lo slave NON vede nessun edge, NON cambiamo i device attuali:
    if not edges:
        log.info(
            "[recompute] slave returned 0 edges for session %s – "
            "keeping current devices: %s",
            session_id,
            ", ".join((sess.get("devices") or {}).keys()) or "none",
        )
        try:
            if SLACK_WEBHOOK_URL:
                send_slack_safe(
                    ":compass: *Recompute skipped* – "
                    f"sessione `{session_id}`, nessun edge dallo slave.\n"
                    f"*Centro*: `({c_lat}, {c_lon})`\n"
                    f"*Raggio*: `{radius_m} m`\n"
                    f"*Device attivi invariati*: "
                    f"`{', '.join((sess.get('devices') or {}).keys()) or 'none'}`"
                )
        except Exception:
            log.exception("[recompute] slack notify failed (skip)")

        return

    # Nuovo set di device visti dallo slave
    new_devices_info: Dict[str, Dict[str, Any]] = {}
    for e in edges:
        dev_id = e.get("device_id")
        if not dev_id:
            continue
        new_devices_info[dev_id] = {
            "lat": e.get("lat"),
            "lon": e.get("lon"),
            "slave_url": slave_url,
        }

    current_devices = sess.get("devices") or {}
    current_ids = set(current_devices.keys())
    new_ids = set(new_devices_info.keys())

    to_add = new_ids - current_ids
    to_remove = current_ids - new_ids

    job_image = sess.get("job_image") or DETECTION_JOB_IMAGE

    # 2) Revert dei device che escono dal raggio
    for dev_id in to_remove:
        info = current_devices.get(dev_id, {})
        dev_slave_url = info.get("slave_url", slave_url)
        log.info(
            "[recompute] device %s non più nel raggio (center=(%s,%s), radius=%sm) → revert",
            dev_id, c_lat, c_lon, radius_m,
        )
        try:
            push_base_workload_to_device(dev_id, dev_slave_url)
        except Exception:
            log.exception("[recompute] failed pushing base workload to %s", dev_id)
        current_devices.pop(dev_id, None)

    # 3) Attivazione dei nuovi device entrati nel raggio
    for dev_id in to_add:
        info = new_devices_info[dev_id]
        dev_lat = info.get("lat")
        dev_lon = info.get("lon")
        dev_slave_url = info.get("slave_url", slave_url)

        log.info(
            "[recompute] device %s entrato nel raggio (center=(%s,%s), radius=%sm) → push detection job",
            dev_id, c_lat, c_lon, radius_m,
        )

        try:
            push_detection_job_to_device(
                device_id=dev_id,
                session_id=session_id,
                job_image=job_image,
                slave_url=dev_slave_url,
            )
        except Exception:
            log.exception("[recompute] failed pushing detection job to %s", dev_id)

        # mettiamo il nuovo device nella mappa in memoria
        current_devices[dev_id] = {
            "lat": dev_lat,
            "lon": dev_lon,
            "slave_url": dev_slave_url,
            "last_event_at": time.time(),
        }

    # 4) Aggiorniamo i device che erano già presenti ma magari con coord aggiornate
    for dev_id in (current_ids & new_ids):
        info = new_devices_info[dev_id]
        entry = current_devices.get(dev_id, {})
        entry["lat"] = info.get("lat", entry.get("lat"))
        entry["lon"] = info.get("lon", entry.get("lon"))
        entry["slave_url"] = info.get("slave_url", entry.get("slave_url"))

    # aggiorniamo la mappa in memoria
    sess["devices"] = current_devices

    # 5) Allineiamo anche il documento Mongo (activated_devices + device_info)
    try:
        db_sess = get_scan_session(session_id) or {}
        db_sess["activated_devices"] = list(current_devices.keys())
        db_sess["device_info"] = {
            dev_id: {
                "lat": info.get("lat"),
                "lon": info.get("lon"),
            }
            for dev_id, info in current_devices.items()
        }
        ttl_seconds = max(
            int(sess.get("hard_timeout_s", SCAN_HARD_TIMEOUT_S)),
            int(sess.get("timeout_s", SCAN_DEFAULT_TIMEOUT_S)),
            60,
        )
        save_scan_session(session_id, db_sess, ttl_seconds=ttl_seconds)
    except Exception:
        log.exception("[recompute] failed syncing session %s with Mongo", session_id)

    # 6) Notifica Slack del nuovo centro e dei device attivi
    try:
        if SLACK_WEBHOOK_URL:
            lines = [
                ":compass: *Recompute raggio & device attivi*",
                f"*Sessione*: `{session_id}`",
                f"*Nuovo centro*: `({c_lat}, {c_lon})`",
                f"*Raggio*: `{radius_m} m`",
                f"*Device attivi nel raggio*: `{', '.join(current_devices.keys()) or 'none'}`",
                f"*Device entrati nel raggio (VRI push)*: `{', '.join(to_add) or 'none'}`",
                f"*Device usciti dal raggio (revert AD)*: `{', '.join(to_remove) or 'none'}`",
            ]
            send_slack_safe("\n".join(lines))
    except Exception:
        log.exception("[recompute] slack notify failed")



#
# ------------------------------------------------------------------------------
# Helpers K8s
# ------------------------------------------------------------------------------
def get_k8s_clients() -> Tuple[client.AppsV1Api, client.CoreV1Api, client.NetworkingV1Api]:
    """
    Ritorna (apps_v1, core_v1, networking_v1) con config in-cluster o locale.
    """
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    return (
        client.AppsV1Api(),
        client.CoreV1Api(),
        client.NetworkingV1Api(),
    )

def host_only(url: str) -> str:
    if not url:
        return ""
    u = url.strip()
    if u.startswith("http://"):
        u = u[len("http://"):]
    elif u.startswith("https://"):
        u = u[len("https://"):]
    return u.split("/", 1)[0]

# ------------------------------------------------------------------------------
# Geodistanza
# ------------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dl / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ------------------------------------------------------------------------------
# Naming helpers
# ------------------------------------------------------------------------------
def _list_zone_gateways(apps_v1: client.AppsV1Api, namespace: str, zone: str) -> List[str]:
    """
    Torna lista di nomi Deployment che matchano lo schema:
      slave-gateway-{zone}-{NN}-deployment
    """
    deployments = apps_v1.list_namespaced_deployment(
        namespace=namespace,
        label_selector=f"zone={zone}"
    )

    pat = re.compile(rf"^slave-gateway-{re.escape(zone)}-(\d+)-deployment$")
    matches = []
    for d in deployments.items:
        dn = d.metadata.name
        if pat.match(dn):
            matches.append(dn)

    return matches


def _next_suffix_for_zone(apps_v1: client.AppsV1Api, namespace: str, zone: str) -> str:
    """
    Trova il prossimo suffisso NN libero in quella zona.
    """
    existing = _list_zone_gateways(apps_v1, namespace, zone)
    pat = re.compile(rf"^slave-gateway-{re.escape(zone)}-(\d+)-deployment$")
    used_nums = []
    for name in existing:
        m = pat.match(name)
        if m:
            used_nums.append(int(m.group(1)))
    if not used_nums:
        return "01"
    return f"{max(used_nums) + 1:02d}"


def _build_gateway_names(zone: str, suffix: str) -> Dict[str, str]:
    """
    Dato zone="oristano", suffix="01" ->
      slave-gateway-oristano-01-{deployment,svc,ing}
      host = slave-gateway-oristano-01.<BASE_DOMAIN>
    """
    base = f"slave-gateway-{zone}-{suffix}"
    return {
        "base": base,
        "deploy_name": f"{base}-deployment",
        "svc_name": f"{base}-svc",
        "ing_name": f"{base}-ing",
        "host": f"{base}.{BASE_DOMAIN}",
    }


# ------------------------------------------------------------------------------
# K8s object builders
# ------------------------------------------------------------------------------
def _make_gateway_deployment(deploy_name: str, zone: str) -> Dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": deploy_name,
            "labels": {
                "app": deploy_name,
                "zone": zone,
            },
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": deploy_name,
                    "zone": zone,
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": deploy_name,
                        "zone": zone,
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "slave-gateway",
                            "image": SLAVE_IMAGE,
                            "imagePullPolicy": "Always",
                            "env": [
                                {"name": "ZONE", "value": zone},
                                {"name": "LOG_LEVEL", "value": "INFO"},
                            ],
                            "ports": [{"containerPort": 8080}],
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080,
                                    "scheme": "HTTP",
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10,
                                "timeoutSeconds": 2,
                                "failureThreshold": 6,
                                "successThreshold": 1,
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080,
                                    "scheme": "HTTP",
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 15,
                                "timeoutSeconds": 2,
                                "failureThreshold": 6,
                                "successThreshold": 1,
                            },
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "64Mi"},
                                "limits": {"cpu": "250m", "memory": "128Mi"},
                            },
                        }
                    ]
                },
            },
        },
    }



def _make_gateway_service(svc_name: str, deploy_name: str, zone: str) -> Dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": svc_name,
            "labels": {
                "app": deploy_name,
                "zone": zone,
            },
        },
        "spec": {
            "type": "ClusterIP",
            "selector": {
                "app": deploy_name,
                "zone": zone,
            },
            "ports": [
                {
                    "name": "http",
                    "port": 80,
                    "targetPort": 8080,
                    "protocol": "TCP",
                }
            ],
        },
    }


def _make_gateway_ingress(ing_name: str, host: str, svc_name: str, zone: str) -> Dict[str, Any]:
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {
            "name": ing_name,
            "labels": {
                "app": svc_name.replace("-svc", "-deployment"),
                "zone": zone,
            },
            "annotations": {
                "nginx.ingress.kubernetes.io/backend-protocol": "HTTP",
                "nginx.ingress.kubernetes.io/ssl-redirect": "false",
                "nginx.ingress.kubernetes.io/proxy-connect-timeout": "180",
                "nginx.ingress.kubernetes.io/proxy-read-timeout": "180",
                "nginx.ingress.kubernetes.io/proxy-send-timeout": "180",
                "nginx.ingress.kubernetes.io/rewrite-target": "/$1",
            },
        },
        "spec": {
            "ingressClassName": "nginx",
            "rules": [
                {
                    "host": host,
                    "http": {
                        "paths": [
                            {
                                "path": "/(.*)",
                                "pathType": "ImplementationSpecific",
                                "backend": {
                                    "service": {
                                        "name": svc_name,
                                        "port": {"number": 80},
                                    }
                                },
                            }
                        ]
                    },
                }
            ],
        },
    }

# ------------------------------------------------------------------------------
# internal helpers for slave lifecycle
# ------------------------------------------------------------------------------
def _delete_gateway_resources(apps_v1, core_v1, net_v1, names: Dict[str, str]):
    """
    Cancella deploy/svc/ingress se esistono.
    """
    deploy_name = names["deploy_name"]
    svc_name = names["svc_name"]
    ing_name = names["ing_name"]

    try:
        apps_v1.delete_namespaced_deployment(
            name=deploy_name,
            namespace=K8S_TARGET_NAMESPACE,
        )
        log.info("[cleanup] deleted deploy %s", deploy_name)
    except ApiException as e:
        if e.status != 404:
            log.warning("[cleanup] can't delete deploy %s: %s", deploy_name, e)

    try:
        core_v1.delete_namespaced_service(
            name=svc_name,
            namespace=K8S_TARGET_NAMESPACE,
        )
        log.info("[cleanup] deleted svc %s", svc_name)
    except ApiException as e:
        if e.status != 404:
            log.warning("[cleanup] can't delete svc %s: %s", svc_name, e)

    try:
        net_v1.delete_namespaced_ingress(
            name=ing_name,
            namespace=K8S_TARGET_NAMESPACE,
        )
        log.info("[cleanup] deleted ing %s", ing_name)
    except ApiException as e:
        if e.status != 404:
            log.warning("[cleanup] can't delete ing %s: %s", ing_name, e)

def check_slave_health(url: str, timeout_s: int = HEALTHCHECK_TIMEOUT) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=timeout_s)
        return r.status_code == 200
    except Exception as e:
        log.warning("slave health check failed for %s: %s", url, e)
        return False

def register_device_to_slave(
    url: str,
    device_id: str,
    lat: float,
    lon: float,
    retries: int = 5,
    delay_s: float = 2.0,
) -> Tuple[int, Optional[dict]]:
    payload = {
        "device_id": device_id,
        "lat": lat,
        "lon": lon,
    }

    last_status = 0
    last_data = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(f"{url}/register", json=payload, timeout=5)
            status = r.status_code
            data = None
            try:
                data = r.json()
            except Exception:
                data = None

            log.info(
                "[register_device_to_slave] attempt=%s url=%s status=%s body=%s",
                attempt, url, status, (r.text or "")[:200],
            )

            # se va bene → stop subito
            if 200 <= status < 300:
                return status, data

            # se 503/502/504 → probabilmente slave non ancora pronto, riprovo
            if status in (502, 503, 504) and attempt < retries:
                time.sleep(delay_s)
                last_status, last_data = status, data
                continue

            # altri errori → non ha senso riprovare
            return status, data

        except Exception as e:
            log.warning(
                "[register_device_to_slave] error calling slave %s (attempt=%s): %s",
                url, attempt, e,
            )
            last_status, last_data = 0, None
            if attempt < retries:
                time.sleep(delay_s)

    # tutte le retry fallite
    return last_status, last_data


def set_device_mode(slave_url: str, device_id: str, mode: str) -> bool:
    """
    Chiama sullo slave:
      POST {slave_url}/device_mode
      Body: { "device_id": "...", "mode": "AD|VRI|MUTED" }
    """
    mode = str(mode).upper()
    if mode not in ("AD", "VRI", "MUTED"):
        log.warning("[device_mode] invalid mode=%s for device=%s", mode, device_id)
        return False

    if not slave_url:
        log.warning("[device_mode] no slave_url for device=%s (mode=%s)", device_id, mode)
        return False

    try:
        url = f"{slave_url.rstrip('/')}/device_mode"
        payload = {"device_id": device_id, "mode": mode}
        log.info("[device_mode] POST %s payload=%s", url, payload)

        r = requests.post(url, json=payload, timeout=5)
        if 200 <= r.status_code < 300:
            log.info(
                "[device_mode] device=%s mode=%s set successfully (status=%s)",
                device_id, mode, r.status_code,
            )
            return True

        log.error(
            "[device_mode] slave responded %s: %s",
            r.status_code,
            (r.text or "")[:200],
        )
        return False
    except Exception as e:
        log.error(
            "[device_mode] error calling slave %s for device %s: %s",
            slave_url, device_id, e,
        )
        return False


# ------------------------------------------------------------------------------
# ensure_slave_gateway
# ------------------------------------------------------------------------------
def ensure_slave_gateway(zone: str) -> Dict[str, Any]:
    """
    Strategia:
      - cerca gateway esistenti in zona
      - se esiste quello con suffix minore:
            se healthy -> riusalo
            se non healthy -> delete & recreate stesso suffix
      - se non ne esistono: crea nuovo suffix
    """
    apps_v1, core_v1, net_v1 = get_k8s_clients()

    existing_deploys = _list_zone_gateways(apps_v1, K8S_TARGET_NAMESPACE, zone)
    existing_deploys.sort()

    if existing_deploys:
        deploy_name = existing_deploys[0]
        suffix = deploy_name.split("-")[-2]
        names = _build_gateway_names(zone, suffix)

        slave_url = f"http://{names['host']}"

        if check_slave_health(slave_url):
            log.info("[ensure_slave_gateway] reusing healthy %s url=%s", deploy_name, slave_url)
            return {
                "name": names["deploy_name"],
                "service": names["svc_name"],
                "host": names["host"],
                "url": slave_url,
            }

        log.warning("[ensure_slave_gateway] %s unhealthy. Recreating...", deploy_name)
        _delete_gateway_resources(apps_v1, core_v1, net_v1, names)
        time.sleep(2)

        dep_body = _make_gateway_deployment(names["deploy_name"], zone)
        apps_v1.create_namespaced_deployment(namespace=K8S_TARGET_NAMESPACE, body=dep_body)

        svc_body = _make_gateway_service(names["svc_name"], names["deploy_name"], zone)
        core_v1.create_namespaced_service(namespace=K8S_TARGET_NAMESPACE, body=svc_body)

        ing_body = _make_gateway_ingress(names["ing_name"], names["host"], names["svc_name"], zone)
        net_v1.create_namespaced_ingress(namespace=K8S_TARGET_NAMESPACE, body=ing_body)

        time.sleep(3)

        return {
            "name": names["deploy_name"],
            "service": names["svc_name"],
            "host": names["host"],
            "url": f"http://{names['host']}",
        }

    # no gateway in zona -> creane uno nuovo
    suffix = _next_suffix_for_zone(apps_v1, K8S_TARGET_NAMESPACE, zone)
    names = _build_gateway_names(zone, suffix)

    dep_body = _make_gateway_deployment(names["deploy_name"], zone)
    apps_v1.create_namespaced_deployment(namespace=K8S_TARGET_NAMESPACE, body=dep_body)

    svc_body = _make_gateway_service(names["svc_name"], names["deploy_name"], zone)
    core_v1.create_namespaced_service(namespace=K8S_TARGET_NAMESPACE, body=svc_body)

    ing_body = _make_gateway_ingress(names["ing_name"], names["host"], names["svc_name"], zone)
    net_v1.create_namespaced_ingress(namespace=K8S_TARGET_NAMESPACE, body=ing_body)

    time.sleep(3)

    return {
        "name": names["deploy_name"],
        "service": names["svc_name"],
        "host": names["host"],
        "url": f"http://{names['host']}",
    }

def get_slave_gateway_url_for_zone(zone: str) -> str:
    info = ensure_slave_gateway(zone)
    return info["url"]


def get_slave_gateway_name_for_zone(zone: str) -> str:
    info = ensure_slave_gateway(zone)
    return info["name"]


def get_slave_gateway_service_for_zone(zone: str) -> str:
    info = ensure_slave_gateway(zone)
    return info["service"]


def slave_url_for_container(full_slave_url: str) -> str:
    # togli lo schema perché i tuoi container lo vogliono così
    if not full_slave_url:
        return ""
    p = urlparse(full_slave_url)
    if p.scheme:
        return p.netloc
    return full_slave_url

def list_all_slave_gateways(namespace: str) -> List[Dict[str, Any]]:
    """
    Ritorna una lista di tutti gli slave gateway noti nel cluster.
    """
    apps_v1, core_v1, net_v1 = get_k8s_clients()

    deployments = apps_v1.list_namespaced_deployment(namespace=namespace).items

    gateways: List[Dict[str, Any]] = []

    for dep in deployments:
        dname = dep.metadata.name or ""
        m = re.match(r"^slave-gateway-([a-z0-9-]+-\d+)-deployment$", dname)
        if not m:
            continue

        zone_plus_suffix = m.group(1)  # es "oristano-01"
        if "-" in zone_plus_suffix:
            zone = "-".join(zone_plus_suffix.split("-")[:-1])
        else:
            zone = zone_plus_suffix

        base_prefix = f"slave-gateway-{zone_plus_suffix}"

        svc_name = f"{base_prefix}-svc"
        ing_name = f"{base_prefix}-ing"

        try:
            ing_obj = net_v1.read_namespaced_ingress(name=ing_name, namespace=namespace)
            host = None
            if ing_obj and ing_obj.spec and ing_obj.spec.rules:
                host = ing_obj.spec.rules[0].host
            if host:
                url = f"http://{host}"
            else:
                url = None
        except ApiException:
            host = None
            url = None

        gateways.append({
            "zone": zone,
            "name": dname,
            "svc": svc_name,
            "ing": ing_name,
            "host": host,
            "url": url,
        })

    return gateways

def query_slave_edges(slave_url: str, lat: float, lon: float, radius_m: float) -> Dict[str, Any]:
    """
    Chiama sullo slave:
      GET {slave_url}/edges_nearby?lat=...&lon=...&radius_m=...
    """
    if not slave_url:
        log.warning("[edges_nearby] no slave_url provided")
        return {"ok": False, "edges": [], "error": "no_url"}

    try:
        url = f"{slave_url}/edges_nearby"
        params = {"lat": lat, "lon": lon, "radius_m": radius_m}
        log.info("[edges_nearby] GET %s params=%r", url, params)

        r = requests.get(url, params=params, timeout=5)

        log.info(
            "[edges_nearby] response status=%s body=%s",
            r.status_code,
            (r.text or "")[:500],
        )

        if r.status_code != 200:
            return {"ok": False, "edges": [], "status": r.status_code, "body": (r.text or "")[:500]}
        data = r.json()
        edges = data.get("edges", [])
        return {"ok": True, "edges": edges, "raw": data}
    except Exception as e:
        log.warning("[edges_nearby] exception: %s", e)
        return {"ok": False, "edges": [], "error": str(e)}



# ------------------------------------------------------------------------------
# IoT Hub (manifest)
# ------------------------------------------------------------------------------
def _parse_iothub_conn_string(conn: str):
    parts = dict(p.split("=", 1) for p in conn.split(";") if "=" in p)
    host = parts.get("HostName")
    policy = parts.get("SharedAccessKeyName")
    key = parts.get("SharedAccessKey")
    if not host or not policy or not key:
        raise ValueError("IOTHUB_CONNECTION_STRING invalid.")
    return host, policy, key


def _build_sas_token(uri: str, key_b64: str, policy_name: str, ttl_secs: int = 3600) -> str:
    expiry = int(time.time()) + ttl_secs
    sign_key = base64.b64decode(key_b64)
    string_to_sign = (quote(uri, safe="") + "\n" + str(expiry)).encode("utf-8")
    sig = base64.b64encode(hmac.new(sign_key, string_to_sign, hashlib.sha256).digest()).decode()
    token = (
        "SharedAccessSignature "
        f"sr={quote(uri, safe='')}&sig={quote(sig, safe='')}&se={expiry}&skn={quote(policy_name, safe='')}"
    )
    return token


def push_manifest_to_iothub(device_id: str, modules_content: dict):
    sdk_err = None
    # prova SDK
    try:
        try:
            from azure.iot.hub import IoTHubRegistryManager  # type: ignore
        except Exception:
            IoTHubRegistryManager = None
        if IoTHubRegistryManager is not None:
            reg = IoTHubRegistryManager(IOTHUB_CONNECTION_STRING)
            reg.apply_configuration_content_on_device(device_id, {"modulesContent": modules_content})
            return True, None
    except Exception as e:
        sdk_err = str(e)

    # fallback REST
    try:
        hub_host, policy, key = _parse_iothub_conn_string(IOTHUB_CONNECTION_STRING)
        device_id_enc = quote(device_id, safe="")
        resource_aud = f"{hub_host}/devices/{device_id_enc}".lower()
        sas = _build_sas_token(resource_aud, key, policy)
        url = f"https://{hub_host}/devices/{device_id_enc}/applyConfigurationContent?api-version=2020-09-30"
        headers = {"Authorization": sas, "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, data=json.dumps({"modulesContent": modules_content}), timeout=20)
        if 200 <= resp.status_code < 300:
            return True, None
        return False, f"REST {resp.status_code}: {resp.text}; {sdk_err or ''}"
    except Exception as e:
        return False, f"REST exception: {e}; {sdk_err or ''}"


def ensure_one_project_module_in_edgeagent(
    device_id: str,
    image: str,
    env: dict,
    module_name: str = None,
):
    """
    Imposta SOLO il workload base (AD) come modulo IoT Edge:
      - modulo base: running, restartPolicy=always
      - NON tocca runtime / systemModules / $edgeHub
    """
    if has_app_context():
        log = current_app.logger
    else:
        # fallback quando siamo in un thread in background, fuori dal contesto Flask
        log = logging.getLogger("master-node")

    if module_name is None:
        module_name = DEFAULT_WORKLOAD_NAME  # es. "myWorkloadModule"

    create_options_workload = {
        "Cmd": ["bash", "DemoAD_start.sh"]
    }

    modules = {
        module_name: {
            "version": "1.0",
            "type": "docker",
            "status": "running",
            "restartPolicy": "always",
            "settings": {
                "image": image,
                "createOptions": json.dumps(create_options_workload),
            },
            "env": {k: {"value": v} for k, v in env.items()},
        }
    }

    # ⚠️ Tocchiamo solo $edgeAgent.properties.desired.modules
    modules_content = {
        "$edgeAgent": {
            "properties.desired": {
                "schemaVersion": "1.1",
                "modules": modules,
            }
        }
    }

    ok, err = push_manifest_to_iothub(device_id, modules_content)
    if not ok:
        log.error("[workload] failed to push manifest to %s: %s", device_id, err)
    else:
        log.info("[workload] manifest pushed to %s", device_id)
        if SLACK_WEBHOOK_URL:
            try:
                send_slack_safe(
                    f":package: *Base workload pushed* → device `{device_id}` "
                    f"(modulo `{module_name}`, immagine `{image}`)"
                )
            except Exception:
                log.exception("[slack] base workload push notify failed")


def ensure_detection_job_module_in_edgeagent(
    device_id: str,
    image: str,
    env: dict,
    module_name: str = "myDetectionJob",
):
    """
    Aggiorna SOLO la sezione modules:
      - DEFAULT_WORKLOAD_NAME -> stopped
      - myDetectionJob        -> running, restartPolicy=never
    Non tocca runtime/systemModules/$edgeHub.
    """
    create_options = {
        "Cmd": ["bash", "DemoReID_start.sh"],
        "Labels": {
            "one.session": env.get("SCAN_SESSION_ID", ""),
            "one.deploy_ts": str(int(time.time())),
        }
    }

    modules = {
        DEFAULT_WORKLOAD_NAME: {
            "version": "1.0",
            "type": "docker",
            "status": "stopped",
            "restartPolicy": "always",
            "settings": {
                "image": DEFAULT_WORKLOAD_IMAGE,
                "createOptions": json.dumps({"Cmd": ["bash", "DemoAD_start.sh"]}),
            },
            "env": {},
        },
        module_name: {
            "version": "1.0",
            "type": "docker",
            "status": "running",
            "restartPolicy": "never",  # job non deve ripartire da solo
            "settings": {
                "image": image,
                "createOptions": json.dumps(create_options),
            },
            "env": {k: {"value": v} for k, v in env.items()},
        },
    }

    modules_content = {
        "$edgeAgent": {
            "properties.desired": {
                "schemaVersion": "1.1",
                "modules": modules,
            }
        }
    }

    ok, err = push_manifest_to_iothub(device_id, modules_content)
    if not ok:
        current_app.logger.error("[detection-job] failed to push manifest to %s: %s", device_id, err)
    else:
        current_app.logger.info("[detection-job] manifest pushed to %s", device_id)


def push_base_workload_manifest(device_id: str, image: str, env: dict):
    """Ripristina SOLO il workload base running e rimuove/ferma il job lato desired."""
    if has_app_context():
        log = current_app.logger
    else:
        log = logging.getLogger("master-node")

    create_options = {"Cmd": ["bash", "DemoAD_start.sh"]}

    env_base = {
        "EDGE_DEVICE_ID": device_id,
        "SLAVE_URL": env.get("SLAVE_URL", ""),
        "SLAVE_TOKEN": env.get("SLAVE_TOKEN", SLAVE_SHARED_TOKEN or ""),
        "MASTER_HTTP_URL": env.get("MASTER_HTTP_URL", ""),
    }

    modules = {
        DEFAULT_WORKLOAD_NAME: {
            "version": "1.0",
            "type": "docker",
            "status": "running",
            "restartPolicy": "always",
            "settings": {
                "image": image,
                "createOptions": json.dumps(create_options),
            },
            "env": {k: {"value": v} for k, v in env_base.items()},
        }
        # NOTA: nessun myDetectionJob qui → viene rimosso dal desired
    }

    modules_content = {
        "$edgeAgent": {
            "properties.desired": {
                "schemaVersion": "1.1",
                "modules": modules,
            }
        }
    }

    ok, err = push_manifest_to_iothub(device_id, modules_content)
    if ok:
        log.info("[revert] base workload back on %s (myWorkloadModule running, job rimosso)", device_id)
    else:
        log.error("[revert] push failed for %s: %s", device_id, err)


def push_base_workload_manifest(device_id: str, image: str):
    """Ripristina il workload base running e rimuove/ferma il job lato desired."""
    if has_app_context():
        log = current_app.logger
    else:
        log = logging.getLogger("master-node")

    create_options = {"Cmd": ["bash", "DemoAD_start.sh"]}

    env_base = {
        "EDGE_DEVICE_ID": device_id,
        "SLAVE_URL": "",
        "SLAVE_TOKEN": SLAVE_SHARED_TOKEN or "",
        "MASTER_HTTP_URL": ""
    }

    modules = {
        DEFAULT_WORKLOAD_NAME: {
            "version": "1.0",
            "type": "docker",
            "status": "running",
            "restartPolicy": "always",
            "settings": {
                "image": image,
                "createOptions": json.dumps(create_options),
            },
            "env": {k: {"value": v} for k, v in env_base.items()},
        }
        # niente myDetectionJob → viene rimosso dal desired
    }

    modules_content = {
        "$edgeAgent": {
            "properties.desired": {
                "schemaVersion": "1.1",
                "modules": modules,
            }
        }
    }

    ok, err = push_manifest_to_iothub(device_id, modules_content)
    if ok:
        log.info("[revert] base workload back on %s", device_id)
        if SLACK_WEBHOOK_URL:
            send_slack_safe(
                f":white_check_mark: Revert completato su `{device_id}` – "
                f"workload ripristinato, job rimosso."
            )
    else:
        log.error("[revert] push failed for %s: %s", device_id, err)
        if SLACK_WEBHOOK_URL:
            send_slack_safe(
                f":x: Revert *fallito* su `{device_id}` – errore push manifest: `{err}`"
            )



def push_detection_job_to_device(device_id: str, session_id: str, job_image: str, slave_url: str):
    """
    Pusha il job di detection sul device e, nell'immediato, mette il device in modalità MUTED
    sullo slave, così eventuali ingest residui di AD vengono ignorati mentre il job parte.

    Quando arrivano i primi eventi /api/v1/scan/report il master imposterà poi la mode=VRI.
    """
    slave_host_only = slave_url_for_container(slave_url)
    env = {
        "EDGE_DEVICE_ID": device_id,
        "SLAVE_URL": slave_host_only,
        # token usato dal master per verificare il job
        "SLAVE_TOKEN": SLAVE_SHARED_TOKEN,
        # alias nel caso il container usi EDGE_TOKEN invece di SLAVE_TOKEN
        "EDGE_TOKEN": SLAVE_SHARED_TOKEN,
        # sessione da riportare a /api/v1/scan/report
        "SCAN_SESSION_ID": session_id,
        # esattamente come quando lo lanciavi a mano:
        #  MASTER_HTTP_URL=128.203.65.69.nip.io
        "MASTER_HTTP_URL": MASTER_BASE_HOST,
    }

    log.info("[scan] pushing detection job to %s with image=%s", device_id, job_image)

    try:
        if SLACK_WEBHOOK_URL:
            send_slack_safe(
                f":runner: *VRI starting* → device `{device_id}` "
                f"session `{session_id}` immagine `{job_image}` "
                f"(slave `{slave_host_only}`)"
            )
    except Exception:
        log.exception("[slack] VRI start notify failed")

    # 1) pubblichiamo il manifest del job (DEFAULT_WORKLOAD_NAME → stopped, job → running)
    ensure_detection_job_module_in_edgeagent(
        device_id=device_id,
        image=job_image,
        env=env,
        module_name=DETECTION_JOB_MODULE,
    )

    # 2) subito dopo il push del job mettiamo il device in MUTED sullo slave:
    #    in questo stato lo slave IGNORA gli /ingest provenienti da AD
    try:
        set_device_mode(slave_url, device_id, "MUTED")
    except Exception:
        log.exception(
            "[scan] failed to set device %s mode MUTED on slave %s",
            device_id, slave_url,
        )



# ------------------------------------------------------------------------------
# scheduler revert
# ------------------------------------------------------------------------------
def schedule_revert_to_workload(device_id: str, slave_url: str, timeout_s: int, job_image: str):
    base_env = {
        "EDGE_DEVICE_ID": device_id,
        "SLAVE_URL": slave_url_for_container(slave_url),
        "SLAVE_TOKEN": SLAVE_SHARED_TOKEN or "",
        "MASTER_HTTP_URL": MASTER_BASE_HOST,
    }

    def _worker():
        try:
            log.info("[scheduler] waiting %ss before reverting %s", timeout_s, device_id)
            time.sleep(max(1, int(timeout_s)))
            push_revert_manifest(
                device_id=device_id,
                base_image=DEFAULT_WORKLOAD_IMAGE,
                base_env=base_env,
                job_image=job_image,
            )

            # dopo il revert "hard", rimettiamo il device in AD sullo slave
            try:
                set_device_mode(slave_url, device_id, "AD")
            except Exception:
                log.exception(
                    "[scheduler] failed to set device %s mode AD on slave %s after revert",
                    device_id, slave_url,
                )

            if SLACK_WEBHOOK_URL:
                send_slack_safe(
                    f":rewind: *Revert eseguito* → device `{device_id}` "
                    f"(AD running, VRI stopped)"
                )
        except Exception as e:
            log.error("[scheduler] error reverting %s: %s", device_id, e)
            if SLACK_WEBHOOK_URL:
                send_slack_safe(f":x: *Revert FAILED* su `{device_id}`: `{e}`")

    threading.Thread(target=_worker, daemon=True).start()



def push_revert_manifest(device_id: str, base_image: str, base_env: dict, job_image: str):
    """
    Manifest di revert "completo":
      - myWorkloadModule  -> running, restartPolicy=always
      - myDetectionJob    -> stopped, restartPolicy=never
    Non manda messaggi Slack, li gestiamo a livello di sessione in /scan/report.
    """
    create_options_workload = {"Cmd": ["bash", "DemoAD_start.sh"]}
    create_options_job = {"Cmd": ["bash", "DemoReID_start.sh"]}

    modules = {
        DEFAULT_WORKLOAD_NAME: {
            "version": "1.0",
            "type": "docker",
            "status": "running",
            "restartPolicy": "always",
            "settings": {
                "image": base_image,
                "createOptions": json.dumps(create_options_workload),
            },
            "env": {k: {"value": v} for k, v in (base_env or {}).items()},
        },
        DETECTION_JOB_MODULE: {
            "version": "1.0",
            "type": "docker",
            "status": "stopped",
            "restartPolicy": "never",
            "settings": {
                "image": job_image,
                "createOptions": json.dumps(create_options_job),
            },
            "env": {},
        },
    }

    modules_content = {
        "$edgeAgent": {
            "properties.desired": {
                "schemaVersion": "1.1",
                "modules": modules,
            }
        }
    }

    ok, err = push_manifest_to_iothub(device_id, modules_content)
    if ok:
        log.info("[revert] base AD back on %s; detection VRI stopped", device_id)
    else:
        log.error("[revert] push failed for %s: %s", device_id, err)




# dopo il push del job e il log "waiting XXXs before reverting ..."
def _revert_later(device_id: str, hard_timeout_s: int, base_image: str, base_env: dict, job_image: str):
    if has_app_context():
        logger = current_app.logger
    else:
        logger = logging.getLogger("master-node")
    try:
        time.sleep(max(1, int(hard_timeout_s)))
        push_revert_manifest(
            device_id=device_id,
            base_image=base_image,
            base_env=base_env,
            job_image=job_image,
        )
    except Exception:
        logger.exception("[scheduler] revert thread crashed")
        if SLACK_WEBHOOK_URL:
            send_slack_safe(":x: *Scheduler crash* nel revert; controllare i log del master.")

def _scan_sessions_gc_loop():
    """
    GC delle sessioni in memoria.

    NOTA:
    - Il revert AD/VRI adesso lo facciamo nel ramo hard_timeout di /api/v1/scan/report
      (quando arrivano eventi dal job dopo la scadenza).
    - Questo loop serve solo a ripulire SCAN_SESSIONS per non far crescere la memoria.
    """
    while True:
        now = time.time()

        for sid in list(SCAN_SESSIONS.keys()):
            sess = SCAN_SESSIONS.get(sid)
            if not sess:
                continue

            hard_deadline = sess.get("hard_expires_at")

            # se non ho hard_deadline o non è ancora passato → niente
            if not hard_deadline or now <= hard_deadline:
                continue

            # se è già stata revertita, la tolgo semplicemente
            if sess.get("reverted"):
                log.info(
                    "[gc] removing already-reverted session %s from memory "
                    "(now=%.0f, deadline=%.0f)",
                    sid,
                    now,
                    hard_deadline,
                )
                SCAN_SESSIONS.pop(sid, None)
                continue

            # altrimenti: è scaduta ma il revert sarà (o è stato) gestito da /scan/report
            log.info(
                "[gc] session %s hard_timeout superato (now=%.0f > deadline=%.0f) – "
                "nessun revert qui, ci pensa /scan/report",
                sid,
                now,
                hard_deadline,
            )
            SCAN_SESSIONS.pop(sid, None)

        time.sleep(5)


# ------------------------------------------------------------------------------
# Email e Slack
# ------------------------------------------------------------------------------
def send_email_notification(to_addr: str, subject: str, body: str) -> bool:
    if not to_addr:
        return False
    try:
        msg = EmailMessage()
        msg["From"] = SMTP_FROM
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.ehlo()
            s.starttls()
            if SMTP_USER and SMTP_PASS:
                s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True
    except Exception as ex:
        log.error("email send failed: %s", ex)
        return False


def send_email_safe(to: str, subject: str, body: str) -> bool:
    log = current_app.logger
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USERNAME")   # es. il tuo account Gmail
    pwd  = os.getenv("SMTP_PASSWORD")   # app password se 2FA
    from_addr = os.getenv("SMTP_FROM", "master@one-project.local")

    try:
        msg = MIMEText(body, _charset="utf-8")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to

        with smtplib.SMTP(host, port, timeout=20) as s:
            s.ehlo()
            try:
                s.starttls()
                s.ehlo()
            except Exception:
                pass
            if user and pwd:
                s.login(user, pwd)
            s.sendmail(from_addr, [to], msg.as_string())
        return True
    except Exception as e:
        log.error("email send failed: %s", e)
        return False

def send_slack_safe(text: str, blocks: Optional[list] = None) -> bool:
    """
    Invia una notifica a Slack via Incoming Webhook.
    Ritorna True/False ma non lancia mai eccezioni.
    """
    # logger “furbo”: usa current_app se c’è, altrimenti il logger globale
    if has_app_context():
        logger = current_app.logger
    else:
        logger = logging.getLogger("master-node")

    url = SLACK_WEBHOOK_URL
    if not url:
        logger.debug("[slack] webhook URL not configured, skipping message")
        return False

    try:
        payload = {"text": text}
        if blocks:
            payload["blocks"] = blocks
        r = requests.post(url, json=payload, timeout=10)
        ok = 200 <= r.status_code < 300
        if not ok:
            logger.error(
                "[slack] webhook returned %s: %s",
                r.status_code,
                (r.text or "")[:300],
            )
        return ok
    except Exception as e:
        logger.error("[slack] send failed: %s", e)
        return False


# ------------------------------------------------------------------------------
# routes
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "role": "master"}), 200


@app.route("/api/v1/bootstrap/edge", methods=["POST"])
@require_auth
def api_bootstrap_edge():
    data = request.get_json(force=True) or {}

    device_id = data.get("device_id")
    if not device_id:
        return jsonify({"ok": False, "error": "missing device_id"}), 400

    # lat / lon
    lat = data.get("lat") or (data.get("coords") or {}).get("lat")
    lon = data.get("lon") or (data.get("coords") or {}).get("lon")

    # 1) zona
    zone = data.get("zone")
    if not zone:
        if lat and lon:
            zone = pick_zone_for_coords(float(lat), float(lon))
        else:
            zone = "default"

    # 2) slave per la zona
    slave_info = ensure_slave_gateway(zone)
    slave_url_full = slave_info.get("url", "")
    slave_host_only = slave_url_for_container(slave_url_full)

    # host del master visto dal container (solo host, niente schema)
    master_host_only = host_only(
        request.headers.get("Host")
        or os.getenv("MASTER_BASE_HOST")
        or BASE_DOMAIN
    )

    # 3) registra il device sullo slave SOLO se abbiamo lat/lon sensate
    if lat is None or lon is None:
        current_app.logger.warning(
            "[bootstrap] missing lat/lon for device %s – "
            "NOT registering on slave (edge_registry resterà vuoto)",
            device_id,
        )
        reg_status, reg_json = 0, {"ok": False, "error": "missing_lat_lon"}
    else:
        reg_status, reg_json = register_device_to_slave(
            slave_url_full,
            device_id,
            float(lat),
            float(lon),
        )

    # ------------------------------------------------------------------
    # 4) workload richiesto (di default myWorkloadModule + wlchar_ad_api_v2_cpu)
    #    --> QUI è la modifica importante
    # ------------------------------------------------------------------
    workload = data.get("workload") or {}

    # nome del modulo
    module_name = (
        workload.get("name")
        or data.get("workload_name")
        or DEFAULT_WORKLOAD_NAME
    )

    # immagine: proviamo varie chiavi, sia dentro "workload" che top-level
    requested_image = (
        workload.get("image_url")
        or workload.get("image")
        or workload.get("imageUrl")
        or data.get("workload_image")
        or data.get("image_url")
        or data.get("image")
        or data.get("imageUrl")
    )

    module_image = requested_image or DEFAULT_WORKLOAD_IMAGE

    current_app.logger.info(
        "[bootstrap] device %s workload name=%s requested_image=%r used_image=%r",
        device_id, module_name, requested_image, module_image
    )
    # ------------------------------------------------------------------
    # FINE BLOCCO MODIFICATO
    # ------------------------------------------------------------------

    # 5) env per il modulo base
    env_for_module = {
        "EDGE_DEVICE_ID": device_id,
        "SLAVE_URL": slave_host_only,       # SENZA http://
        "SLAVE_TOKEN": SLAVE_SHARED_TOKEN,
        "MASTER_HTTP_URL": master_host_only # SENZA schema
    }

    extra_env = workload.get("env")
    if isinstance(extra_env, dict):
        for k, v in extra_env.items():
            # non sovrascrivo le chiavi core
            env_for_module.setdefault(k, str(v))

    # 6) push manifest per workload base
    ensure_one_project_module_in_edgeagent(
        device_id=device_id,
        image=module_image,
        env=env_for_module,
        module_name=module_name,
    )

    # 6-bis) Rimettiamo il device in AD sullo slave (riattiva /ingest)
    try:
        set_device_mode(slave_url_full, device_id, "AD")
        current_app.logger.info(
            "[bootstrap] device %s rimesso in mode AD sullo slave %s",
            device_id, slave_url_full,
        )
    except Exception:
        current_app.logger.exception(
            "[bootstrap] failed to set device %s mode AD on slave %s",
            device_id, slave_url_full,
        )

    # 7) notifica Slack bootstrap
    try:
        if SLACK_WEBHOOK_URL:
            send_slack_safe(
                f":electric_plug: *Bootstrap edge* per device `{device_id}` "
                f"zona `{zone}` → slave `{slave_info.get('host','?')}` "
                f"modulo `{module_name}` immagine `{module_image}`\n"
                f"*Coords*: `({lat}, {lon})`"
            )
    except Exception:
        current_app.logger.exception("[slack] bootstrap notify failed")

    # 8) risposta
    return jsonify({
        "ok": True,
        "error": None,
        "device_id": device_id,
        "zone": zone,
        "slave": slave_info,
        "slave_register_status": [reg_status, reg_json],
        "workload": {
            "name": module_name,
            "image_url": module_image,
            "env": env_for_module,
        },
    }), 200




@app.post("/api/v1/scan/start")
@require_bearer
def api_scan_start():
    log = current_app.logger

    try:
        data = request.get_json(silent=True) or {}

        # Input essenziali
        try:
            lat = float(data["lat"])
            lon = float(data["lon"])
        except Exception:
            return jsonify({"ok": False, "error": "lat/lon required"}), 400

        radius_m     = float(data.get("radius_m", 300))
        timeout_s    = int(data.get("timeout_s", SCAN_DEFAULT_TIMEOUT_S))
        hard_timeout = int(data.get("hard_timeout_s", SCAN_HARD_TIMEOUT_S))
        job_image    = data.get("job_image") or DETECTION_JOB_IMAGE
        notify_email = data.get("notify_email") or ""

        # Zona auto
        zone = data.get("zone")
        if not zone:
            zone = pick_zone_for_coords(lat, lon)

        # Slave per zona
        slave = ensure_slave_gateway(zone)
        slave_url = slave.get("url", "")

        # --- CHIAMATA ALLO SLAVE ---
        edges_resp = query_slave_edges(slave_url, lat, lon, radius_m)

        if not edges_resp.get("ok"):
            log.warning(
                "[scan-start] query_slave_edges failed zone=%s url=%s resp=%r",
                zone, slave_url, edges_resp
            )
            # risposta leggibile invece di 500
            return jsonify({
                "ok": False,
                "error": "slave_edges_unavailable",
                "details": edges_resp,
            }), 502

        edges = edges_resp.get("edges", []) or []

        activated_devices = [e["device_id"] for e in edges if e.get("device_id")]

        # ❌ NESSUN FALLBACK QUI
        # se non ci sono device, rispondiamo comunque in modo pulito
        if not activated_devices:
            log.info(
                "[scan-start] no devices in radius (zone=%s center=(%s,%s) radius=%s)",
                zone, lat, lon, radius_m
            )
            return jsonify({
                "ok": True,
                "session_id": None,
                "center": {"lat": lat, "lon": lon},
                "radius_m": radius_m,
                "timeout_s": timeout_s,
                "hard_timeout_s": hard_timeout,
                "notify_email": notify_email,
                "job_image": job_image,
                "activated_devices": [],
                "message": "no devices in radius",
            }), 200

        # Info per ogni device trovato
        devices_info = {}
        for e in edges:
            dev_id = e.get("device_id")
            if not dev_id:
                continue
            devices_info[dev_id] = {
                "lat": e.get("lat", lat),
                "lon": e.get("lon", lon),
                "slave_url": slave_url,
            }

        # --- COSTRUZIONE SESSIONE ---
        session_id = f"scan:{int(time.time())}"
        session_data = {
            "center": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "timeout_s": timeout_s,
            "hard_timeout_s": hard_timeout,
            "notify_email": notify_email,
            "zone": zone,
            "job_image": job_image,
            "activated_devices": activated_devices,
            "device_info": devices_info,
        }

        # Persistenza su Mongo
        ttl = max(hard_timeout, timeout_s, 60)
        save_scan_session(session_id, session_data, ttl_seconds=ttl)
        log.info("[scan-start] session saved id=%s ttl=%ss", session_id, ttl)

        # Registriamo anche in memoria per GC “mobile”
        register_scan_session_in_memory(session_id, session_data, devices_info, slave_url)

        # Notifica Slack "Scan started"
        try:
            if SLACK_WEBHOOK_URL:
                lines = [
                    ":satellite: *Scan started*",
                    f"*Sessione*: `{session_id}`",
                    f"*Zona*: `{zone}` (slave `{slave.get('host','?')}`)",
                    f"*Centro iniziale*: `({lat}, {lon})`",
                    f"*Raggio*: `{radius_m} m`",
                    f"*Device nel raggio*: `{', '.join(activated_devices)}`",
                ]
                send_slack_safe("\n".join(lines))
        except Exception:
            log.exception("[scan-start] slack notify failed")

         # Push job + scheduler revert
        for did in activated_devices:
            push_detection_job_to_device(did, session_id, job_image=job_image, slave_url=slave_url)
            # schedule_revert_to_workload(did, slave_url, hard_timeout, job_image=job_image)

        # --- RISPOSTA OK ---
        return jsonify({
            "ok": True,
            "session_id": session_id,
            "center": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "timeout_s": timeout_s,
            "hard_timeout_s": hard_timeout,
            "notify_email": notify_email,
            "job_image": job_image,
            "activated_devices": activated_devices,
        }), 200

    except Exception as e:
        log.exception("[scan-start] unhandled error in /api/v1/scan/start")
        return jsonify({
            "ok": False,
            "error": f"internal_error: {e.__class__.__name__}",
            "message": str(e),
        }), 500


from datetime import datetime  # ce l'hai già sopra, va bene anche così

@app.route("/api/v1/scan/report", methods=["POST"])
@app.route("/api/v1/scan/report/", methods=["POST"])
@require_edge_auth
def api_scan_report():
    log = current_app.logger

    # 1) log grezzo per debug
    raw_body = request.get_data(as_text=True)
    log.warning("[scan-report] raw body=%s", raw_body)

    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        log.exception("[scan-report] invalid json")
        return jsonify({"ok": False, "error": "invalid json"}), 400

    # 2) normalizzazione campi
    session_id = (
        payload.get("session_id")
        or payload.get("SCAN_SESSION_ID")
        or payload.get("session")
        or payload.get("sessionId")
    )

    device_id = (
        payload.get("device_id")
        or payload.get("EDGE_DEVICE_ID")
        or payload.get("device")
        or payload.get("deviceId")
    )

    event = (
        payload.get("event")
        or payload.get("event_type")
        or payload.get("anomaly_type")
        or "detection"
    )

    zone = (
        payload.get("zone")
        or payload.get("slave_zone")
        or payload.get("province")
    )

    details = payload.get("details")
    if details is None:
        details = dict(payload)

    ts = payload.get("timestamp") or payload.get("time") or payload.get("ts")

    # 3) se manca session_id/device_id → non blocchiamo, ma mettiamo "unknown"
    if not session_id:
        log.warning("[scan-report] missing session_id in payload, marking as 'unknown'")
        session_id = "unknown"

    if not device_id:
        log.warning("[scan-report] missing device_id in payload, marking as 'unknown'")
        device_id = "unknown"

    now_ts = time.time()

    # 4) carichiamo la sessione COMPLETA da Mongo (doc + data)
    sess = None
    doc = None
    hard_deadline = None

    try:
        if session_id != "unknown":
            doc = scan_sessions.find_one({"session_id": session_id})
            if doc:
                sess = (doc or {}).get("data") or {}
    except Exception:
        log.exception("[scan-report] Mongo get failed")

    # 4bis) se NON esiste più nemmeno su Mongo → evento davvero "stale"
    if session_id != "unknown" and doc is None:
        log.info(
            "[scan-report] stale event per session_id=%s device=%s – "
            "nessun log Slack/email, solo ack (sessione non trovata su Mongo)",
            session_id,
            device_id,
        )
        return jsonify({
            "ok": True,
            "received": True,
            "stored": False,
            "stale": True,
            "reason": "no_session",
        }), 200

    # 4ter) controllo versione IN MEMORIA (hard_expires_at ha priorità)
    mem_sess = None
    if session_id != "unknown":
        mem_sess = SCAN_SESSIONS.get(session_id)

    if mem_sess and mem_sess.get("hard_expires_at"):
        hard_deadline = mem_sess["hard_expires_at"]
    else:
        # fallback: ricavo la deadline da Mongo (created_at + hard_timeout_s)
        created_at = doc.get("created_at")
        if created_at:
            try:
                created_ts = created_at.timestamp()
            except AttributeError:
                created_ts = float(created_at)
            hard_timeout_s = int((sess or {}).get("hard_timeout_s", SCAN_HARD_TIMEOUT_S))
            hard_deadline = created_ts + hard_timeout_s

    # 4quater) se è passato l'hard timeout → evento stale, revert + solo ack
    if hard_deadline is not None and now_ts > hard_deadline:
        # se la sessione è già stata revertita, non rifaccio nulla
        already_reverted = False
        if sess and sess.get("reverted"):
            already_reverted = True
        if mem_sess and mem_sess.get("reverted"):
            already_reverted = True

        if already_reverted:
            log.info(
                "[scan-report] stale event dopo revert già eseguito per session_id=%s device=%s",
                session_id,
                device_id,
            )
            return jsonify({
                "ok": True,
                "received": True,
                "stored": False,
                "stale": True,
                "reason": "hard_timeout_already_reverted",
            }), 200

        log.info(
            "[scan-report] stale event per session_id=%s device=%s – "
            "hard timeout scaduto (now=%.0f, deadline=%.0f), avvio revert (AD running, VRI stopped)",
            session_id,
            device_id,
            now_ts,
            hard_deadline,
        )

        # segno la sessione come revertita (Mongo + memoria)
        if sess is not None:
            sess["reverted"] = True
            try:
                ttl_seconds = max(
                    int(sess.get("hard_timeout_s", SCAN_HARD_TIMEOUT_S)),
                    int(sess.get("timeout_s", SCAN_DEFAULT_TIMEOUT_S)),
                    60,
                )
                save_scan_session(session_id, sess, ttl_seconds=ttl_seconds)
            except Exception:
                log.exception("[scan-report] failed to mark session %s as reverted", session_id)

        if mem_sess is not None:
            mem_sess["reverted"] = True

        # Ricavo i device da revertire
        devices_to_revert: Dict[str, Dict[str, Any]] = {}
        if mem_sess:
            devices_to_revert = mem_sess.get("devices") or {}
        else:
            dev_info = (sess or {}).get("device_info") or {}
            for did in (sess or {}).get("activated_devices", []):
                info = dev_info.get(did, {})
                devices_to_revert[did] = {
                    "lat": info.get("lat"),
                    "lon": info.get("lon"),
                    "slave_url": "",
                }

        device_ids = list(devices_to_revert.keys())
        job_image = (sess or {}).get("job_image") or DETECTION_JOB_IMAGE

        for did, info in devices_to_revert.items():
            slave_url = info.get("slave_url", "")
            base_env = {
                "EDGE_DEVICE_ID": did,
                "SLAVE_URL": slave_url_for_container(slave_url),
                "SLAVE_TOKEN": SLAVE_SHARED_TOKEN or "",
                "MASTER_HTTP_URL": MASTER_BASE_HOST,
            }

            try:
                # manifest: AD running, VRI stopped
                push_revert_manifest(
                    device_id=did,
                    base_image=DEFAULT_WORKLOAD_IMAGE,
                    base_env=base_env,
                    job_image=job_image,
                )

                # rimettiamo il device in AD sullo slave (se conosciamo lo slave_url)
                if slave_url:
                    try:
                        set_device_mode(slave_url, did, "AD")
                    except Exception:
                        log.exception(
                            "[scan-report] failed to set device %s mode AD on slave %s dopo revert",
                            did, slave_url,
                        )
            except Exception:
                log.exception("[scan-report] revert (manifest) failed for device %s", did)

        # 🔔 QUI lo Slack, UNA VOLTA per la sessione
        try:
            if SLACK_WEBHOOK_URL and device_ids:
                send_slack_safe(
                    f":white_check_mark: *Revert completato* sessione `{session_id}` "
                    f"device: `{', '.join(device_ids)}` – AD running, VRI stopped."
                )
        except Exception:
            log.exception("[scan-report] revert slack notify failed")

        # Pulisco la sessione in memoria per evitare revert multipli
        if session_id in SCAN_SESSIONS:
            SCAN_SESSIONS.pop(session_id, None)

        return jsonify({
            "ok": True,
            "received": True,
            "stored": False,
            "stale": True,
            "reason": "hard_timeout",
        }), 200



    # ---- DA QUI IN GIÙ: sessione ANCORA ATTIVA → recenter, store, slack, ecc. ----

    # 5) gestione in memoria (recenter + timeout mobile + recompute)

    # 5a) Se la sessione non è in memoria ma esiste su Mongo, la ricostruiamo
    if not mem_sess and sess:
        zone = sess.get("zone")
        slave_url = ""
        if zone:
            try:
                slave_info = ensure_slave_gateway(zone)
                slave_url = slave_info.get("url", "")
            except Exception:
                log.exception(
                    "[scan-report] ensure_slave_gateway failed while rebuilding mem session %s",
                    session_id,
                )

        devices_info: Dict[str, Dict[str, Any]] = {}
        for dev_id, info in (sess.get("device_info") or {}).items():
            devices_info[dev_id] = {
                "lat": info.get("lat"),
                "lon": info.get("lon"),
                "slave_url": slave_url,
            }

        register_scan_session_in_memory(session_id, sess, devices_info, slave_url)
        mem_sess = SCAN_SESSIONS.get(session_id)

        log.info(
            "[scan-report] rebuilt in-memory session %s from Mongo (devices=%s)",
            session_id,
            ", ".join((devices_info or {}).keys()) or "none",
        )

    # 5b) Se ora abbiamo mem_sess (normale o ricostruita) facciamo recenter + recompute
    if mem_sess:
        timeout_s = int(mem_sess.get("timeout_s", SCAN_DEFAULT_TIMEOUT_S))
        dev_entry = (mem_sess.get("devices") or {}).get(device_id)

        if dev_entry:
            dev_entry["last_event_at"] = now_ts
            dev_lat = dev_entry.get("lat")
            dev_lon = dev_entry.get("lon")

            if dev_lat is not None and dev_lon is not None:
                mem_sess["center"] = {"lat": dev_lat, "lon": dev_lon}
                log.info(
                    "[scan-report] recentered session %s on device %s at (%s, %s)",
                    session_id, device_id, dev_lat, dev_lon
                )

                try:
                    sess["center"] = {"lat": dev_lat, "lon": dev_lon}
                    ttl_seconds = max(
                        int(mem_sess.get("hard_timeout_s", SCAN_HARD_TIMEOUT_S)),
                        timeout_s,
                        60,
                    )
                    save_scan_session(session_id, sess, ttl_seconds=ttl_seconds)
                except Exception:
                    log.exception("[scan-report] failed updating session center in Mongo")

        mem_sess["expires_at"] = now_ts + timeout_s
        log.debug(
            "[scan-report] session %s sliding timeout extended to +%ss (until %.0f)",
            session_id, timeout_s, mem_sess["expires_at"]
        )

        recompute_active_devices_for_session(session_id)
    else:
        # qui siamo solo nel caso veramente strano: sessione non in memoria
        # e neppure su Mongo (ma questo caso in pratica non dovrebbe arrivare,
        # perché sopra abbiamo già gestito i "stale").
        log.debug(
            "[scan-report] session %s non presente in SCAN_SESSIONS e non ricostruibile – "
            "salvo solo evento + notifiche",
            session_id,
        )

    # 6) Salviamo l'evento in DB
    stored = True
    try:
        store_scan_event(
            session_id=session_id,
            device_id=device_id,
            zone=zone or (sess or {}).get("zone"),
            event=event,
            details=details,
            timestamp=ts,
        )
    except Exception:
        stored = False
        log.exception("[scan-report] store_scan_event failed")

    # 7) Notifica email (solo se la sessione esiste e ha notify_email)
    email_sent = False
    try:
        notify_email = (sess or {}).get("notify_email")
        if notify_email:
            dev_info = ((sess or {}).get("device_info", {}) or {}).get(device_id, {})
            dev_lat  = dev_info.get("lat")
            dev_lon  = dev_info.get("lon")
            subject = f"[ONE] Evento {event} – sessione {session_id}"
            body = (
                f"Evento: {event}\n"
                f"Sessione: {session_id}\n"
                f"Device: {device_id}\n"
                f"Zona: {zone or (sess or {}).get('zone')}\n"
                f"Timestamp: {ts}\n"
                f"Dettagli: {json.dumps(details, ensure_ascii=False)}\n"
                + (f"Posizione device: {dev_lat}, {dev_lon}\n" if (dev_lat and dev_lon) else "")
            )
            email_sent = send_email_safe(notify_email, subject, body)
    except Exception:
        log.exception("[scan-report] email notify failed")

    # 8) Notifica Slack
    try:
        if SLACK_WEBHOOK_URL:
            dev_info = ((sess or {}).get("device_info", {}) or {}).get(device_id, {})
            dev_lat  = dev_info.get("lat")
            dev_lon  = dev_info.get("lon")

            lines = [
                f"*Evento*: `{event}`",
                f"*Sessione*: `{session_id}`",
                f"*Device*: `{device_id}`",
            ]

            if ts:
                lines.append(f"*Timestamp*: `{ts}`")
            if details:
                lines.append(f"*Dettagli*: ```{json.dumps(details, ensure_ascii=False)}```")
            if dev_lat and dev_lon:
                lines.append(f"*Posizione device*: `{dev_lat}, {dev_lon}`")

            slack_text = "\n".join(lines)
            send_slack_safe(slack_text)
    except Exception:
        log.exception("[scan-report] slack notify failed")

    return jsonify({
        "ok": True,
        "received": True,
        "stored": stored,
        "email_sent": email_sent,
        "stale": False,
    }), 200




# avvio thread GC
threading.Thread(target=_scan_sessions_gc_loop, daemon=True).start()
