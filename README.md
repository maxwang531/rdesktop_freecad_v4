# rdesktop_freecad_v3
rdesktop-Startbefehl:
docker run -d --name=rdesktop --privileged  -e PUID=1000 -e PGID=1000 -e TZ=Europe/London -p 3401:3389 -v /var/run/docker.sock:/var/run/docker.sock -v ~/home/mathi/dockervolume/cncdemo/rdesktop_freecad_v3/:/config --shm-size="2gb" --restart unless-stopped maxwang531/rdesktop_freecad_stable_baselines3:v3

Konto: abc, Passwort: abc

SsdNet-Startbefehl:
docker run --gpus "device=1" -e CUDA_VISIBLE_DEVICES=1 -i -t --name demo_feature --privileged -v ~/home/mathi/dockervolume/cncdemo/rdesktop_freecad_v3/:/docker_demo/src maxwang531/ssdnet_cuda10:latest /bin/bash

download.sh:
Laden das erforderliche Trainingsmodell für SsdNet herunter

Schritt1.sh:
Modellvorverarbeitung

Schritt2.sh:
Wird im SsdNet-Container ausgeführt. 
Befehl:
cd docker_demo/src/mnt/SsdNet
conda activate py36
python SsdNet.py

Schritt3.sh:
Codegenerierung für grundlegende Merkmale.

Schritt4.sh:
Generierung von Bearbeitungscodes für Freiformflächen durch Reinforcement Learning

Schritt5.sh:
Benutzerdefiniertes Reinforcement Learning Training

profile.sh:
Rohprofilfräsen

simulation.sh:
Freiformflächen-Ergebnissimulation


