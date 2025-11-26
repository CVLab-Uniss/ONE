import os
import time
import math
import logging
import json
import requests
from typing import Any, Dict, List
from flask import Flask, request, jsonify

# =========================
# CONFIG
# =========================
PORT = int(os.getenv("PORT", "8080"))
ZONE = os.getenv("ZONE", "unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
SLACK_WORKLOAD_WEBHOOK_URL = os.getenv("SLACK_WORKLOAD_WEBHOOK_URL", "").strip()


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [slave-gateway %(zone)s] %(message)s".replace("%(zone)s", ZONE),
)
logger = logging.getLogger("slave-gateway")

app = Flask(__name__)

# in-memory registry { device_id: {...} }
edge_registry: Dict[str, Dict[str, Any]] = {}


def send_slack_safe(text: str) -> bool:
    """
    Invia un messaggio sul canale Slack dedicato ai log del workload.
    Non lancia mai eccezioni, ritorna solo True/False.
    """
    url = SLACK_WORKLOAD_WEBHOOK_URL
    if not url:
        # webhook non configurato → nessun invio
        return False
    try:
        payload = {"text": text}
        r = requests.post(url, json=payload, timeout=10)
        ok = 200 <= r.status_code < 300
        if not ok:
            logger.error(
                "Slack workload webhook returned %s: %s",
                r.status_code,
                (r.text or "")[:300],
            )
        return ok
    except Exception as e:
        logger.error("Slack workload send failed: %s", e)
        return False


# =========================
# UTILS GEO
# =========================
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distanza in metri tra due coordinate geografiche.
    Implementazione super-basica e robusta.
    """
    R = 6371000.0  # raggio medio Terra in metri

    # cast esplicito a float, giusto per evitare stranezze con stringhe
    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Sanity check a runtime (solo log)
try:
    d_test = haversine_m(39.905, 8.592, 39.905, 8.592)
    logger.info("[haversine_sanity] dist_same_point=%s (atteso ~0)", d_test)
except Exception as e:
    logger.error("[haversine_sanity] failed: %s", e)

# =========================
# ENDPOINTS
# =========================

@app.get("/health")
def health():
    return jsonify({"ok": True, "zone": ZONE}), 200


# ---- INGEST generico -------------------------------------------------
def _ingest_common(payload: Dict[str, Any]):
    """
    Log unico che usiamo da /ingest e /api/v1/ingest.
    Se il device è in modalità MUTED o VRI, ignoriamo il payload.
    Se nel payload non c'è device_id ma abbiamo UN SOLO device registrato,
    attribuiamo il payload a quel device (fallback per ambienti 1-device).
    """
    # 1) cerchiamo di capire da che device arriva il payload
    device_id = (
        payload.get("device_id")
        or payload.get("EDGE_DEVICE_ID")
        or payload.get("edge_id")
        or payload.get("edge_device_id")
    )

    # Fallback se non c'è device_id nel payload
    if not device_id:
        if len(edge_registry) == 1:
            device_id = next(iter(edge_registry.keys()))
            logger.info(
                "[ingest] payload senza device_id, lo attribuisco a device=%s (fallback 1 device)",
                device_id,
            )
        elif len(edge_registry) > 1:
            logger.info(
                "[ingest] payload senza device_id e %s device registrati – "
                "non posso applicare mute per-mode, accetto comunque",
                len(edge_registry),
            )

    if device_id:
        info = edge_registry.get(device_id)
        if info:
            # aggiorniamo sempre il last_seen
            info["last_seen"] = time.time()
            mode = info.get("mode", "AD")

            # TRATTIAMO SIA MUTED CHE VRI COME "MUTI" LATO INGEST
            if mode in ("MUTED", "VRI"):
                logger.info(
                    "[ingest] device_id=%s mode=%s – payload ignorato",
                    device_id,
                    mode,
                )
                return jsonify({
                    "ok": False,
                    "ignored": True,
                    "reason": "muted" if mode == "MUTED" else "vri",
                    "device_id": device_id,
                    "mode": mode,
                }), 200
        else:
            logger.info(
                "[ingest] payload da device non registrato device_id=%s – nessuna mode, accetto",
                device_id,
            )

    # 2) se non è MUTED/VRI, o non abbiamo device_id → log normale
    logger.info("[ingest-debug] payload keys=%s", list(payload.keys()))
    logger.info("[ingest-debug] computed device_id=%r", device_id)
    logger.info("[ingest-debug] registry_keys=%s", list(edge_registry.keys()))
    # Log “pulito” del payload che arriva dal workload
    logger.info("[workload-ingest] payload=%s", json.dumps(payload, ensure_ascii=False))

    if SLACK_WORKLOAD_WEBHOOK_URL:
        try:
            text_lines = [
                ":camera: *Workload ingest*",
                f"*Zona*: `{ZONE}`",
                f"*Payload*:",
                "```" + json.dumps(payload, ensure_ascii=False)[:1500] + "```",
            ]
            text = "\n".join(text_lines)
            

            # 👇 LOG identico a quello che invii su Slack
            logger.info("[slack-workload] %s", text)

            send_slack_safe(text)
        except Exception:
            logger.exception("Slack workload ingest notify failed")


    return jsonify({"ok": True, "received": True}), 200


@app.post("/ingest")
def ingest():
    """
    Riceve risultati inference dagli edge.
    Alcune tue immagini mandano application/json, altre a volte text/plain.
    Qui normalizziamo.
    """
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        # prova a parsare uguale
        try:
            payload = request.get_json(force=True, silent=True) or {}
        except Exception:
            payload = {"raw": request.data.decode("utf-8", errors="ignore")}

    return _ingest_common(payload)


@app.post("/api/v1/ingest")
def ingest_v1():
    """
    Versione namespaced, fa la stessa cosa.
    """
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        try:
            payload = request.get_json(force=True, silent=True) or {}
        except Exception:
            payload = {"raw": request.data.decode("utf-8", errors="ignore")}

    return _ingest_common(payload)


@app.post("/register")
def register():
    """
    Chiamato dal master al bootstrap dell'edge.
    Body JSON:
    {
      "device_id": "edgeDevice01",
      "lat": 39.90,
      "lon": 8.59
    }
    Salva in memoria: device_id -> posizione, slave_id (=questa ZONE), last_seen, mode
    """
    if not request.is_json:
        return jsonify({"ok": False, "error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    lat = data.get("lat")
    lon = data.get("lon")

    if device_id is None or lat is None or lon is None:
        return jsonify({"ok": False, "error": "missing device_id/lat/lon"}), 400

    # se il device esiste già, preserviamo la mode corrente
    existing = edge_registry.get(device_id) or {}
    mode = existing.get("mode", "AD")

    edge_registry[device_id] = {
        "device_id": device_id,
        "lat": float(lat),
        "lon": float(lon),
        "slave_id": ZONE,
        "last_seen": time.time(),
        "mode": mode,  # AD / VRI / MUTED
    }

    logger.info(
        "REGISTER device_id=%s slave=%s lat=%s lon=%s mode=%s registry_id=%s count=%s",
        device_id, ZONE, lat, lon, mode, id(edge_registry), len(edge_registry),
    )

    return jsonify({"ok": True}), 200


@app.post("/device_mode")
def set_device_mode():
    """
    Permette di impostare la modalità di un device:
      - AD    : workload normale, ingest attivo
      - VRI   : job di scan attivo, ingest ignorato
      - MUTED : ingest ignorato (mute manuale)
    """
    if not request.is_json:
        return jsonify({"ok": False, "error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    mode_raw = data.get("mode")

    if not device_id or not mode_raw:
        return jsonify({"ok": False, "error": "missing device_id/mode"}), 400

    mode = str(mode_raw).upper()
    if mode not in ("AD", "VRI", "MUTED"):
        return jsonify({"ok": False, "error": "invalid mode, allowed: AD/VRI/MUTED"}), 400

    # Recupero / creo info del device
    info = edge_registry.get(device_id) or {
        "device_id": device_id,
        "lat": None,
        "lon": None,
        "slave_id": ZONE,
        "last_seen": time.time(),
    }

    old_mode = info.get("mode", "AD")
    info["mode"] = mode
    info["last_seen"] = time.time()
    edge_registry[device_id] = info

    # Se la modalità è la stessa → non mandiamo Slack
    if old_mode == mode:
        logger.info(
            "[device_mode] device_id=%s mode già %s – niente cambio (lat=%s lon=%s)",
            device_id, mode, info.get("lat"), info.get("lon"),
        )
        return jsonify({"ok": True, "device_id": device_id, "mode": mode, "unchanged": True}), 200

    # Cambio mode effettivo
    logger.info(
        "[device_mode] device_id=%s set mode=%s (lat=%s lon=%s)",
        device_id, mode, info.get("lat"), info.get("lon"),
    )

    # Slack (solo se cambia)
    if SLACK_WORKLOAD_WEBHOOK_URL:
        try:
            text = (
                f":satellite: *device_mode* – device `{device_id}` "
                f"→ mode `{mode}` (zone `{ZONE}`)"
            )
            send_slack_safe(text)
        except Exception:
            logger.exception("Slack device_mode notify failed")

    return jsonify({"ok": True, "device_id": device_id, "mode": mode}), 200




def _edges_nearby_core(q_lat: float, q_lon: float, radius_m: float):
    """logica condivisa per trovare gli edge in raggio"""
    results: List[Dict[str, Any]] = []

    logger.info(
        "[edges_nearby_core] START q_lat=%s q_lon=%s radius_m=%s registry_id=%s count=%s keys=%s",
        q_lat, q_lon, radius_m, id(edge_registry),
        len(edge_registry), list(edge_registry.keys()),
    )

    for dev_id, info in edge_registry.items():
        dlat = info["lat"]
        dlon = info["lon"]
        d = haversine_m(q_lat, q_lon, dlat, dlon)

        logger.info(
            "[edges_nearby_core] check dev=%s dev_lat=%s dev_lon=%s dist=%s <= radius=%s ?",
            dev_id, dlat, dlon, d, radius_m,
        )

        if d <= radius_m:
            results.append({
                "device_id": dev_id,
                "slave_id": info["slave_id"],
                "lat": dlat,
                "lon": dlon,
                "distance_m": round(d, 3),
                "last_seen": info["last_seen"],
                # volendo potresti riportare anche la mode per debug:
                # "mode": info.get("mode"),
            })

    logger.info(
        "[edges_nearby_core] DONE results_count=%s",
        len(results),
    )

    return {
        "ok": True,
        "center": {"lat": q_lat, "lon": q_lon},
        "radius_m": radius_m,
        "count": len(results),
        "edges": results,
    }


@app.get("/edges_nearby")
def edges_nearby():
    """
    versione "vecchia" senza /api/v1
    /edges_nearby?lat=...&lon=...&radius_m=300
    """
    logger.info(
        "[edges_nearby] registry_id=%s count=%s keys=%s",
        id(edge_registry),
        len(edge_registry),
        list(edge_registry.keys()),
    )

    try:
        q_lat = float(request.args.get("lat"))
        q_lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid or missing lat/lon"}), 400

    radius_m = request.args.get("radius_m")
    if radius_m is None:
        radius_m = 300.0
    else:
        try:
            radius_m = float(radius_m)
        except ValueError:
            return jsonify({"ok": False, "error": "invalid radius_m"}), 400

    out = _edges_nearby_core(q_lat, q_lon, float(radius_m))
    return jsonify(out), 200


@app.get("/api/v1/edges_nearby")
def api_edges_nearby():
    """
    Restituisce i device che questo slave conosce entro un certo raggio.
    Esempio:
      GET /api/v1/edges_nearby?lat=39.905&lon=8.592&radius_km=0.3
    """
    try:
        q_lat = float(request.args.get("lat", ""))
        q_lon = float(request.args.get("lon", ""))
    except ValueError:
        return jsonify({"ok": False, "error": "invalid lat/lon"}), 400

    try:
        radius_km = float(request.args.get("radius_km", "0.3"))
    except ValueError:
        radius_km = 0.3

    radius_m = radius_km * 1000.0

    edges = []
    for dev_id, info in edge_registry.items():
        dlat = info.get("lat")
        dlon = info.get("lon")
        if dlat is None or dlon is None:
            continue

        dist_m = haversine_m(q_lat, q_lon, dlat, dlon)
        if dist_m <= radius_m:
            edges.append({
                "device_id": dev_id,
                "lat": dlat,
                "lon": dlon,
                "distance_m": round(dist_m, 3),
            })

    return jsonify({
        "ok": True,
        "query": {
            "lat": q_lat,
            "lon": q_lon,
            "radius_m": radius_m,
        },
        "count": len(edges),
        "edges": edges,
    }), 200


@app.get("/debug/registry")
def debug_registry():
    return jsonify({
        "zone": ZONE,
        "count": len(edge_registry),
        "edges": list(edge_registry.values()),
        "registry_id": id(edge_registry),
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
