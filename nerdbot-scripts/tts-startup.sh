#!/bin/bash
# Run the TTS server
# TTS_MODEL_PATH = "tts_models/en/ljspeech/speedy-speech"
# TTS_MODEL_PATH = "tts_models/en/ljspeech/glow-tts"
# TTS_VOCODER_PATH = "tts_models/en/ljspeech/hifigan"

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
sleep 30
python -m TTS.server.server --model_name tts_models/en/ljspeech/speedy-speech 2>&1 | tee ~/nerdbot.log &
