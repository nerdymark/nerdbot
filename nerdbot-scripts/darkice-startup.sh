#!/bin/bash

# Find the Anker device card number
CARD_NUM=$(aplay -l | grep "Anker" | awk -F'card ' '{print $2}' | cut -d':' -f1)

if [ -z "$CARD_NUM" ]; then
    echo "Error: Anker device not found"
    exit 1
fi

# Choose config file based on card number
if [ "$CARD_NUM" = "0" ]; then
    CONFIG_FILE="/home/mark/darkice/darkice.cfg"
elif [ "$CARD_NUM" = "2" ]; then
    CONFIG_FILE="/home/mark/darkice/darkice-2-0.cfg"
else
    echo "Error: Anker device on unexpected card number $CARD_NUM"
    exit 1
fi

echo "Using Anker device on card $CARD_NUM with config $CONFIG_FILE"
/usr/local/bin/darkice -c "$CONFIG_FILE"