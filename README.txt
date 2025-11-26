

Overview

The system consists of three main components:

Master Node → the global orchestrator

Slave Gateways → cloud entry points that communicate with Edge devices and master

Edge Devices (IoT Edge nodes) → the deployed IoT devices 

The Master communicates with Azure IoT Hub, manages deployments, and decides which container (module) should run on each Edge device.
The Slaves are “local” gateways that receive standard data from their nearby edges.
Each Edge Device runs Azure IoT Edge and executes the workloads deployed through IoT Hub.





1. Master Node
1.1 Deployment

Runs inside Kubernetes (namespace: dev)

Service name: master-node-app-service (LoadBalancer)

Public hostname: 128.203.65.69.nip.io

Internally serves a Flask web application (Gunicorn, port 5000 → exposed as 80)

1.2 Responsibilities

Tracks all Slave Gateways
Uses the Kubernetes API to list available slave services and their zones.
(list_all_slave_gateways(K8S_TARGET_NAMESPACE))

Bootstraps new Edge Devices
The route POST /api/v1/bootstrap/edge receives registration data from an edge node, selects the nearest slave, and pushes a manifest to IoT Hub with the base module (myWorkloadModule), including its SLAVE_URL.

Queries all slaves for nearby devices
Using GET /api/v1/edges_nearby?lat=...&lon=...&radius_m=..., the master gathers which edges are within a specific geographic radius.

Starts a dynamic “scan session”
Route POST /api/v1/scan/start launches a temporary detection (VRI task) on all edges within a given area, replacing their base module (AD task) with a detection one (myDetectionJob).

Handles scan reports
Route POST /api/v1/scan/report is used by Edge devices running myDetectionJob to report results directly to the Master.
When the master receives a report, it recenters the scan area on that device, redeploys the detection job to edges inside the new area, and restores the base workload on those outside it.

1.3 Main API Routes
Route	Description
POST /api/v1/bootstrap/edge	Registers a new edge device and assigns it to the nearest slave. Pushes its first deployment via IoT Hub.
GET /api/v1/edges_nearby	Aggregates from all slaves the edges located near a given lat/lon within a radius.
POST /api/v1/scan/start	Initiates a scan session. Finds edges in the area, replaces their workload with the detection job, and starts a session timer.
POST /api/v1/scan/report	Called by edges running the detection job to send results directly to the master. The master then updates the area and deployments accordingly.

Example: Start a scan session

curl -X POST \
  -H "Host: 128.203.65.69.nip.io" \
  -H "Authorization: Bearer MySecureToken1234" \
  -H "Content-Type: application/json" \
  -d '{"lat":39.905,"lon":8.592,"radius_m":300,"timeout_s":180}' \
  http://$MASTER_IP/api/v1/scan/start

The master responds with a session_id and the list of activated devices.

1.4 How module deployment works

The master interacts with Azure IoT Hub via the same logic used during bootstrap:

Default workload (AD task)
Deployed using push_base_workload_to_device(device_id, slave_url)

Runs the module myWorkloadModule 

status: running

restartPolicy: always

Env vars: EDGE_DEVICE_ID, SLAVE_URL, LOG_LEVEL

Communicates with the Slave Gateway

Detection job (VRI task)
Deployed using push_detection_job_to_device(device_id, session_id)

Runs the module myDetectionJob

status: running

restartPolicy: never

Env vars:

EDGE_DEVICE_ID

SCAN_SESSION_ID

MASTER_REPORT_URL (without http:// — the container appends it when sending requests)

Communicates directly with the Master

These helper functions call:

push_manifest_to_iothub(device_id, modules_content)


which sends the manifest through IoT Hub’s REST API, updating the desired properties.





2. Slave Gateway
2.1 Deployment

Runs inside Kubernetes (namespace: dev)

Several per region/zone, e.g.:

slave-gateway-oristano-01-svc, slave-gateway-oristano-02-svc, etc.

slave-gateway-roma-01-service

etc.

Each slave exposes a local HTTP endpoint.

2.2 Responsibilities

Registers new Edge devices
When a device completes its bootstrap, it starts posting REGISTER messages to its assigned slave (SLAVE_URL).

Receives standard telemetry / domain data
Edges running myWorkloadModule post their results here.

Responds to Master discovery requests
When the master calls /api/v1/edges_nearby, each slave returns the list of registered edges in that area.

Essentially, the slave is the local registry and ingestion point for its area.





3. Edge Device (Azure IoT Edge Node)
3.1 Structure

An edge device typically runs:

edgeAgent
edgeHub
myWorkloadModule
myDetectionJob   (only during scans)


edgeAgent and edgeHub are system modules

The application modules are dynamically assigned by the master through IoT Hub.

3.2 Expected behavior

Must be registered in the same IoT Hub as the master

Must accept deployment manifests (applyConfigurationContent)

Reads environment variables passed by the master to determine:

- Who to communicate with (Slave or Master)

- What session it belongs to

- What device ID to identify itself

Environment variables per mode
Mode	Variables	Destination
myWorkloadModule (AD)	EDGE_DEVICE_ID, SLAVE_URL, LOG_LEVEL	Sends HTTP data to Slave
myDetectionJob (VRI)	EDGE_DEVICE_ID, SCAN_SESSION_ID, MASTER_REPORT_URL	Sends HTTP results to Master

3.3 Configuration
Setting up the Environment and Installing Ubuntu for Azure IoT Edge

Before installing the Azure IoT Edge runtime on your device, you must first register it within your IoT Hub.
This process creates a secure identity for your computer in Azure and generates the connection string used to link your PC to the hub.

Summary of steps:

Create the IoT Hub on Azure (Phase 1).

Register your device in the IoT Hub.

Obtain the connection string for that device.

Use the connection string to install and configure the IoT Edge runtime on your PC.

If you skip step 2, you will not have the connection string and the installation will fail.


Using Ubuntu or a Virtual Machine (VirtualBox) (if necessary on Windows environment)

VirtualBox is free and works even on Windows Home.

Step 1: Download and Install VirtualBox

Go to: https://www.virtualbox.org/wiki/Downloads

Download VirtualBox for Windows hosts and install it.

Step 2: Download Ubuntu 22.04 LTS

Server edition: https://ubuntu.com/download/server

Desktop edition: https://ubuntu.com/download/desktop

The Server edition is recommended because it is lighter and more suitable for IoT Edge.

Step 3: Create the Virtual Machine in VirtualBox

Open VirtualBox → click New

Name: Ubuntu-IoTEdge

Type: Linux, Version: Ubuntu (64-bit)

RAM: at least 4096 MB

Disk: create a virtual disk of 32 GB or more

In Settings → Storage, attach the Ubuntu ISO file as an optical drive.

Start the VM and follow the Ubuntu installation process.

Connecting the Ubuntu ISO

If you have not already downloaded it, get the ISO file for Ubuntu 22.04 LTS:
https://ubuntu.com/download/server

Example file name: ubuntu-22.04.5-live-server-amd64.iso.

Open VirtualBox → select your VM (Ubuntu-IoTEdge) → click Settings → Storage.
You’ll see something like:

Controller: IDE
  [Empty]


Click [Empty], then on the right click the CD icon → Choose a disk file... → select the downloaded ubuntu-22.04...iso file → click OK.

Checking Boot Order

Go to Settings → System → Motherboard, and make sure:

Optical Drive is the first boot device.

Hard Disk is the second boot device.

Then start the VM — it should boot from the virtual DVD (the ISO file) and begin Ubuntu installation.

Installing Ubuntu

Follow the standard installation steps:

Language: English (or Italian if preferred)

Choose base server installation (for the Server edition).

Set username and password.

Select “Use entire disk” for virtual disk.

Wait for installation to finish and reboot the VM.


Once Ubuntu has started and you’re at the terminal (XXXX@ubuntu:~$), continue with:

Installing Moby (Docker-compatible runtime).

Installing and configuring Azure IoT Edge.

Connecting the VM to your IoT Hub.

Updating Ubuntu
sudo apt update && sudo apt upgrade -y    # updates and upgrades all packages
sudo apt install ca-certificates curl gnupg lsb-release -y    # installs base utilities

Installing Moby (Docker-compatible for IoT Edge)

Azure IoT Edge requires Moby, not Docker Desktop.

curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | sudo tee /etc/apt/sources.list.d/microsoft-prod.list    # adds Microsoft repository
curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null    # imports Microsoft GPG key
sudo apt update
sudo apt install -y azure-cli jq    # installs Azure CLI and jq utility
sudo apt install moby-engine moby-cli -y    # installs Moby engine and CLI
sudo systemctl status docker    # check if Docker service is active

You should see active (running).
To confirm Docker functionality, run:

sudo docker run hello-world    # verifies Moby/Docker installation

Installing Azure IoT Edge Runtime
sudo apt install aziot-edge -y    # installs Azure IoT Edge runtime

Configuring Azure IoT Edge and Deploying Modules

Now you need the IoT Edge device connection string (from your Azure IoT Hub).

In the Azure portal:

Go to your IoT Hub.

Open the IoT Edge section.

Select your device → Connection Keys.

Copy the Primary Connection String.

Return to your Ubuntu VM and configure the IoT Edge runtime with this command:

sudo iotedge config mp --connection-string "YOUR_DEVICE_CONNECTION_STRING"    # replace with your actual connection string


Apply the configuration:

sudo iotedge config apply    # applies IoT Edge configuration


You should see:

Azure IoT Edge has been configured successfully!
Restarting service...
Done.

Checking IoT Edge Status
sudo systemctl status aziot-edged    # shows IoT Edge service status
sudo iotedge system status           # displays IoT Edge system info
sudo iotedge list                    # lists running modules

If everything is OK, you should see something like:

NAME             STATUS    DESCRIPTION
edgeAgent        running   mcr.microsoft.com/azureiotedge-agent:1.5

(Optional) check logs:

sudo iotedge system logs             # view system logs
sudo iotedge logs edgeAgent          # view edgeAgent logs

Bootstrapping the Workload and Cluster Communication Setup

Pull the container image for the workload module:

sudo docker pull pietroruiu/one-project:wlchar_ad_api_v2_cpu    # downloads the AD workload image


Create a configuration file for the Edge device:

sudo nano edge-bootstrap.json    # open text editor to create JSON file


Paste the following content:

{
  "device_id": "edgeDevice01",
  "zone": "oristano",
  "latitude": 39.905,
  "longitude": 8.592,
  "workload_image": "pietroruiu/one-project:wlchar_ad_api_v2_cpu",
  "workload_name": "myWorkloadModule"
}


Use this file as-is for edgeDevice01.
If you later change the device or its location, update only device_id and coordinates (latitude, longitude, or zone).

Bootstrapping the Device with the Master Node
export MASTER_IP=128.203.65.69
echo "MASTER_IP=$MASTER_IP"
curl -X POST -H "Host: 128.203.65.69.nip.io" \
     -H "Authorization: Bearer MySecureToken1234" \
     -H "Content-Type: application/json" \
     --data-binary @edge-bootstrap.json \
     http://$MASTER_IP/api/v1/bootstrap/edge


→ Sends a bootstrap request to the master node with device info.

Checking on the Cluster Side

On the master node, verify that the corresponding slave gateway received the request (e.g., oristano-01 gateway):

kubectl -n dev get pods | Select-String "slave-gateway-oristano"    # locate the slave pod
kubectl -n dev logs <slave-pod-name> --tail=200                     # view last 200 log lines

If the slave gateway is receiving requests, you should see inbound HTTP activity.





4. Azure IoT Hub
Acts as message broker and device registry:

Registers Edge devices

Stores and distributes manifests (desired properties)

Synchronizes module state

The Master communicates with IoT Hub via:

POST https://<iot-hub>.azure-devices.net/devices/<device_id>/applyConfigurationContent?api-version=2020-09-30

Authenticated using the IoT Hub connection string (not the device string).






5. Full End-to-End Bootstrap Flow

Step 1 – Request to Master Node

curl -X POST \
  -H "Authorization: Bearer MySecureToken1234" \
  -H "Content-Type: application/json" \
  -d '{
        "device_id": "edgeDevice01",
        "latitude": XX.XXX,
        "longitude": YY.YYY,
        "workload_image": "pietroruiu/one-project:wlchar_ad_api_v2_cpu"
      }' \
  "http://$MASTER_IP/api/v1/bootstrap/edge"


→ Sends the device registration payload to the master node.

Step 2 – Master Receives and Validates

The FastAPI service on the master:

Validates the authorization token

Verifies that the zone exists

Determines which slave gateway handles that zone

Temporarily registers the device in memory or Redis

Creates a mapping:
edgeDevice01 ↔ slave-gateway-oristano-01

Step 3 – Master Calls the Slave Gateway

Internal call from master to slave:

POST http://slave-gateway-oristano-01:8080/api/v1/register_device


Payload:

{
  "device_id": "edgeDevice01",
  "zone": "oristano",
  "lat": 39.905,
  "lon": 8.592
}


Response:

{"ok": true, "slave": "oristano"}

Slave logs show:

[slave-gateway oristano] REGISTER device_id=edgeDevice01 slave=oristano lat=39.905 lon=8.592

Step 4 – Master Builds the IoT Edge Manifest

The master dynamically generates a deployment manifest:

{
  "modulesContent": {
    "$edgeAgent": {
      "properties.desired": {
        "modules": {
          "myWorkloadModule": {
            "settings": {
              "image": "pietroruiu/one-project:wlchar_ad_api_v2_cpu",
              "createOptions": "{}"
            },
            "type": "docker",
            "status": "running",
            "restartPolicy": "always",
            "env": {
              "SLAVE_URL": { "value": "http://slave-gateway-oristano-01:8080/api/v1/ingest" },
              "SLAVE_TOKEN": { "value": "MySecureToken1234" },
              "ZONE": { "value": "oristano" }
            }
          }
        }
      }
    },
    "$edgeHub": { "properties.desired": {} }
  }
}


→ Represents the full IoT Edge manifest built on-the-fly by the master.

Step 5 – Master Pushes Manifest to Azure IoT Hub

Executed internally using SDK or REST API:

POST https://<iot-hub>.azure-devices.net/devices/edgeDevice01/applyConfigurationContent?api-version=2020-09-30

Headers:

Authorization: SharedAccessSignature sr=<IoTHubURL>&sig=...
Content-Type: application/json

Body: JSON manifest.
If successful, IoT Hub replies:

{"ok": true, "iotedge_deployment": [true, null]}


Step 6 – Edge Device Synchronizes Manifest

On the Edge device:

EdgeAgent detects new desired properties in the device twin

Pulls the specified Docker image

Starts the workload module

Check with:

sudo iotedge list    # verify modules running on the edge device


Example output:

edgeAgent         running
edgeHub           running
myWorkloadModule  running  pietroruiu/one-project:wlchar_ad_api_v2_cpu

Step 7 – Workload Sends Data to Slave Gateway

When the module starts, it uses the environment variables:

SLAVE_URL=http://slave-gateway-oristano-01:8080/api/v1/ingest
SLAVE_TOKEN=MySecureToken1234
ZONE=oristano

It begins sending JSON payloads like:

{
  "slave_node_ip": "slave-gateway-oristano-01",
  "anomaly_type": 2,
  "timestamp": "2025-06-03 15:25:12",
  "vehicle_brand": "VW",
  "vehicle_model": "GOLF5",
  "vehicle_color": "white"
}

Slave logs display:

INFO [slave-gateway oristano] Ingest payload (zone=oristano): {...}

Step 8 – End of Cycle

At this stage:

The Master’s bootstrap task is complete.

Slave gateways receive data payloads.

Edge devices stay synchronized with IoT Hub automatically.






5. VRI task
Step-by-step flow

User initiates a scan session

Master receives /api/v1/scan/start with a position and radius.

Master identifies relevant devices

Queries all slaves to find edge devices within the given area.

Master deploys detection job

Replaces the normal module with myDetectionJob on those edges.

Edges perform detection

Each job runs independently and reports results via:

POST http://128.203.65.69.nip.io/api/v1/scan/report


First report triggers a recenter

The reporting edge becomes the new scan center

Master finds the new set of nearby devices

Devices that fall outside the new area revert to the base workload

Devices inside (old or new) keep or start the detection job

Timeout behavior

If no report arrives after the timeout (e.g., 180 seconds),
all devices return to the base workload (myWorkloadModule).


VRI Task Call and Dynamic Device Management

You can start a VRI scan task by calling the master node API:

CALL FOR THE VRI TASK (not from the device itself)
From Ubuntu

curl -X POST "http://128.203.65.69.nip.io/api/v1/scan/start" \
-H "Authorization: Bearer MySecureToken1234" \
-H "Content-Type: application/json" \
-d '{ "lat": 39.905, "lon": 8.592, "radius_m": 300, "timeout_s": 180, "hard_timeout_s": 240, "job_image": "pietroruiu/one-project:wlchar_reid_api_v2_cpu", "notify_email": "your@email.com" }'

From Windows PowerShell

$uri = "http://128.203.65.69.nip.io/api/v1/scan/start"
$token = "MySecureToken1234" # your ADMIN_TOKEN

$body = @{
    lat = 39.905
    lon = 8.592
    radius_m = 300
    timeout_s = 180
    hard_timeout_s = 240
    job_image = "pietroruiu/one-project:wlchar_reid_api_v2_cpu"
    notify_email = "your@email.com"
} | ConvertTo-Json

$headers = @{ "Authorization" = "Bearer $token" }

Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -ContentType "application/json" -Body $body



By calling POST /api/v1/scan/start only once, the Master:


Finds the devices within the radius (edges_nearby),


Stops myWorkloadModule (AD),


Starts myDetectionJob (VRI) on those devices,


Waits for results.


When one node replies:


It becomes the new center of the area,


The Master updates the active devices (stops those outside the new area and starts those inside), sends an email to the address specified in the request,


The timer restarts.


If no response arrives within timeout_s seconds:


All devices automatically return to myWorkloadModule (AD).



SLACK LINK TO VIEW MASTER LOGS IN REAL TIME

https://uniss-one.slack.com/archives/C09SV5XS08Z

SLACK LINK TO VIEW RESULTS RECEIVED BY THE SLAVE IN REAL TIME

https://uniss-one.slack.com/archives/C09SLL5A030


→ Sends a scan request to the master node to trigger the VRI task.

What Happens When You Trigger /api/v1/scan/start

When you call the endpoint once, the master performs these operations:

Finds all Edge devices within the specified radius (edges_nearby).

Stops the currently running workload (myWorkloadModule, i.e., AD).

Starts a new detection workload (myDetectionJob, i.e., VRI) on those devices.

Waits for results from the active nodes.

Behavior When Devices Respond

The first node that responds becomes the new center of the active zone.

The master updates the list of active devices:

Stops workloads on devices outside the radius.

Starts workloads on devices within range.

An email notification is sent to the address specified in the request (notify_email).

The timer restarts for continuous monitoring.

Behavior When No Response Arrives

If no node responds within 3 minutes (timeout_s = 180):

All devices automatically revert to running the default workload (myWorkloadModule, AD mode).

At this point, the full IoT Edge architecture — from VM creation to module orchestration, workload bootstrap, and dynamic scanning — is complete and fully operational.
All components (Master Node, Slave Gateway, IoT Hub, and Edge Devices) remain synchronized through Azure IoT Edge deployment and the Kubernetes-based backend.





6. Global Configuration Variables (Master)

At the top of app.py, define:

# Base workload
DEFAULT_WORKLOAD_NAME = "myWorkloadModule"
DEFAULT_WORKLOAD_IMAGE = "pietroruiu/one-project:wlchar_ad_api_v2_cpu"

# Detection job
DETECTION_JOB_MODULE = "myDetectionJob"
DETECTION_JOB_IMAGE = "pietroruiu/one-project:wlchar_reid_api_v2_cpu"

# Public host of the master (no http://)
MASTER_BASE_HOST = "128.203.65.69.nip.io"

# Report endpoint used by detection jobs
MASTER_REPORT_URL = f"{MASTER_BASE_HOST}/api/v1/scan/report"

# Default timeout for scan sessions
SCAN_DEFAULT_TIMEOUT_S = 180


This ensures every job launched by the master automatically knows where to report results, without hardcoding full URLs.




7. Summary
Component	Role	Communicates With
Master Node	Central orchestrator (handles scan logic, IoT Hub manifests, and global state)	Azure IoT Hub, Slaves, Edges
Slave Gateway	Local registry and ingestion service	Master, Edges (base workload)
Edge Device	Executes workloads (myWorkloadModule or myDetectionJob)	Slave or Master depending on mode