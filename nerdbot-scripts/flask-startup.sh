#!/bin/bash
PATH=/home/mark/.local/bin:/home/mark/.nvm/versions/node/v20.17.0/bin:/home/mark/.cargo/bin:/home/mark/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games
cd ~/nerdbot-backend
source setup_env.sh
python -m flask_server.server 2>&1 | tee ~/nerdbot.log &
