#!/bin/bash
# NerdBot Service Management Script

SERVICES="nerdbot-fan nerdbot-flask nerdbot-ui nerdbot-tts nerdbot-joystick nerdbot-darkice"

case "$1" in
    start)
        echo "Starting all NerdBot services..."
        sudo systemctl start $SERVICES
        ;;
    stop)
        echo "Stopping all NerdBot services..."
        sudo systemctl stop $SERVICES
        ;;
    restart)
        echo "Restarting all NerdBot services..."
        sudo systemctl restart $SERVICES
        ;;
    status)
        echo "Status of all NerdBot services:"
        sudo systemctl status $SERVICES --no-pager
        ;;
    enable)
        echo "Enabling all NerdBot services for auto-start..."
        sudo systemctl enable $SERVICES
        ;;
    disable)
        echo "Disabling auto-start for all NerdBot services..."
        sudo systemctl disable $SERVICES
        ;;
    logs)
        SERVICE=${2:-nerdbot-flask}
        echo "Showing logs for $SERVICE (press Ctrl+C to exit)..."
        sudo journalctl -u $SERVICE -f
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|enable|disable|logs [service-name]}"
        echo ""
        echo "Available services:"
        for svc in $SERVICES; do
            echo "  - $svc"
        done
        echo ""
        echo "Examples:"
        echo "  $0 start              # Start all services"
        echo "  $0 status             # Show status of all services"
        echo "  $0 logs nerdbot-flask # Show live logs for flask service"
        exit 1
        ;;
esac