cd /config/mnt/SsdNet/weights
sudo wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QrBqICuPpTw1lHjcFm4f2iMbzTfr-IRc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QrBqICuPpTw1lHjcFm4f2iMbzTfr-IRc" -O VOC.pth && rm -rf /tmp/cookies.txt


