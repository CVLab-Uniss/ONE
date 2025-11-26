# Overview

The system consists of three main components:

- **Master Node** → the global orchestrator  
- **Slave Gateways** → cloud entry points that communicate with Edge devices and the master  
- **Edge Devices (IoT Edge nodes)** → the deployed IoT devices

The Master communicates with Azure IoT Hub, manages deployments, and decides which container (module) should run on each Edge device. The Slaves are “local” gateways that receive standard data from nearby edges. Each Edge Device runs Azure IoT Edge and executes the workloads deployed through IoT Hub.

---

# 1. Master Node

## 1.1 Deployment
Runs inside Kubernetes (namespace: `dev`).

- **Service name:** `master-node-app-service` (LoadBalancer)
- **Public hostname:** `128.203.65.69.nip.io`
- Internally serves a Flask web application (Gunicorn, port 5000 → exposed as 80)

## 1.2 Responsibilities

### Tracks Slave Gateways
Uses the Kubernetes API to list available slave services and their zones.

### Bootstraps New Edge Devices
`POST /api/v1/bootstrap/edge` registers the device, selects nearest slave, and pushes its first manifest to IoT Hub.

### Queries Nearby Devices
`GET /api/v1/edges_nearby` retrieves devices within a geographic radius.

### Starts Dynamic Scan Sessions (VRI)
`POST /api/v1/scan/start` replaces the base workload (AD) with a detection job (VRI) in the scan area.

### Handles Scan Reports
`POST /api/v1/scan/report` recenters the scan area and updates active devices.

## 1.3 Main API Routes

| Route | Description |
|-------|-------------|
| **POST /api/v1/bootstrap/edge** | Registers a device, assigns nearest slave, deploys base workload. |
| **GET /api/v1/edges_nearby** | Returns nearby devices aggregated from all slaves. |
| **POST /api/v1/scan/start** | Starts a scan, deploys detection workload. |
| **POST /api/v1/scan/report** | Devices send detection results here. |

## 1.4 Module Deployment Logic

### Base Workload (AD)
- Module: `myWorkloadModule`
- `restartPolicy: always`
- Communicates with Slave Gateway.

### Detection Job (VRI)
- Module: `myDetectionJob`
- `restartPolicy: never`
- Communicates directly with Master.

Both use:
```
push_manifest_to_iothub(device_id, modules_content)
```

## 1.5 Git Repository Structure

```
infra/
  aks/
    bicep-or-terraform/
k8s/
  base/
    namespace-dev.yaml
    ingress-nginx/...
    mongo/...
    redis/...
  apps/
    master/
      deployment.yaml
      service.yaml
      ingress.yaml
      configmap.yaml
      secret.example.yaml
    slave/
      README.md
docs/
  bootstrap-edge.md
  scan-api.md
```

### Replication Procedure
```
git clone <repo>
edit *-config.yaml and *-secret.yaml
kubectl apply -f k8s/base
kubectl apply -f k8s/apps/master
```

---

# 2. Environment Setup

## Azure Resources
```
az group create -n rg-one-dev -l westeurope
az iot hub create -n camera-node-agent -g rg-one-dev --sku S1
az aks create -g rg-one-dev -n aks-one-dev --node-count 1 --generate-ssh-keys
az aks get-credentials -g rg-one-dev -n aks-one-dev
```

## Kubernetes Bootstrap
```
kubectl create namespace dev
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx --namespace dev
```

Get ingress IP: `kubectl -n dev get svc`

Configure BASE_DOMAIN accordingly.

## Deploy Base Services
```
kubectl -n dev apply -f k8s/base/mongo/
kubectl -n dev apply -f k8s/base/redis/
```

## Deploy Master
```
kubectl -n dev apply -f k8s/apps/master/
kubectl -n dev get pods
```

---

# 3. Slave Gateway

## 3.1 Deployment
Runs in `dev` namespace, multiple per zone.

## 3.2 Responsibilities
- Registers devices
- Receives telemetry from AD nodes
- Responds to Master discovery queries

## 3.3 Reusability
Slaves are created automatically by the Master using:
```
ensure_slave_gateway(zone)
```

### Requirements
- Correct `SLAVE_IMAGE`
- `BASE_DOMAIN` resolves e.g.:
```
slave-gateway-<zone>-01.<BASE_DOMAIN>
```

---

# 4. Edge Device (Azure IoT Edge Node)

## 4.1 Structure
Runs:
- `edgeAgent`
- `edgeHub`
- `myWorkloadModule`
- `myDetectionJob` (during scans)

## 4.2 Behavior
Devices must:
- Be registered in IoT Hub
- Accept manifests
- Use env vars for routing

### Environment Variables
| Mode | Variables |
|------|-----------|
| AD | `EDGE_DEVICE_ID`, `SLAVE_URL`, `LOG_LEVEL` |
| VRI | `EDGE_DEVICE_ID`, `SCAN_SESSION_ID`, `MASTER_REPORT_URL` |

## 4.3 Installation & Configuration
Includes instructions for Ubuntu installation, VirtualBox, IoT Edge setup, Moby installation, device connection string, and module bootstrap.

---

# 5. Full Bootstrap Flow

1. Device sends bootstrap payload to Master.
2. Master validates and assigns slave.
3. Master registers device on slave.
4. Master builds & pushes manifest to IoT Hub.
5. Edge receives updated twin, deploys modules.
6. AD module sends telemetry to slave.

---

# 6. VRI (Detection) Task

- Master receives `/scan/start` request.
- Finds nearby devices.
- Deploys detection job.
- Devices report results to Master.
- First report recenters the scan.
- Timeout restores AD workload.

Curl example:
```
curl -X POST "http://128.203.65.69.nip.io/api/v1/scan/start" \
  -H "Authorization: Bearer MySecureToken1234" \
  -H "Content-Type: application/json" \
  -d '{"lat":39.905,"lon":8.592,"radius_m":300,"timeout_s":180}'
```

---

# 7. Master Global Variables
```
DEFAULT_WORKLOAD_NAME = "myWorkloadModule"
DEFAULT_WORKLOAD_IMAGE = "pietroruiu/one-project:wlchar_ad_api_v2_cpu"
DETECTION_JOB_MODULE = "myDetectionJob"
DETECTION_JOB_IMAGE = "pietroruiu/one-project:wlchar_reid_api_v2_cpu"
MASTER_BASE_HOST = "128.203.65.69.nip.io"
MASTER_REPORT_URL = f"{MASTER_BASE_HOST}/api/v1/scan/report"
SCAN_DEFAULT_TIMEOUT_S = 180
```

---

# 8. Summary

| Component | Role | Communicates With |
|-----------|-------|-------------------|
| **Master Node** | Orchestrates workloads, manages IoT Hub manifests, scan logic | IoT Hub, Slaves, Edges |
| **Slave Gateway** | Local registry & ingestion service | Master, Edges |
| **Edge Device** | Runs workloads (AD / VRI modes) | Slave or Master |

