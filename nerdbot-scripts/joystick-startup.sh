#!/bin/bash
# Integrated Joystick Control Service Startup Script for NerdBot

cd /home/mark/nerdbot-backend

# Source the environment
source setup_env.sh

# Start the integrated joystick service (zero-latency direct control)
echo "Starting integrated joystick control service..."
python integrated_joystick_service.py

echo "Integrated joystick service stopped."