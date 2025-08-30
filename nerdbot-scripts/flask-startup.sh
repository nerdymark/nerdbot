#!/bin/bash
PATH=/home/mark/.local/bin:/home/mark/.nvm/versions/node/v20.17.0/bin:/home/mark/.cargo/bin:/home/mark/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games

# Set audio volume to 75% to prevent USB voltage drops
# Try to set volume on the Anker device specifically
if aplay -l | grep -q "Anker"; then
    # Get the card number for Anker device
    CARD_NUM=$(aplay -l | grep "Anker" | sed -n 's/card \([0-9]\):.*/\1/p')
    if [ -n "$CARD_NUM" ]; then
        # Set volume to 75% using amixer (95 out of 127 range)
        amixer -c $CARD_NUM cset numid=3 95 2>/dev/null
        echo "Set audio volume to 75% on card $CARD_NUM (Anker device)"
    fi
fi

# Also set system-wide volume as fallback
amixer sset Master 75% 2>/dev/null || amixer sset PCM 75% 2>/dev/null

cd ~/nerdbot-backend
source setup_env.sh
python -m flask_server.server 2>&1 | tee ~/nerdbot.log &
