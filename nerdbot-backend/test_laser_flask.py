#!/usr/bin/env python3
"""
Minimal Flask server to test laser control
"""
import logging
from flask import Flask, jsonify
from laser_control.laser_control import LaserControl

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Initialize laser control FIRST before anything else
laser_control = LaserControl()
logging.info(f"Laser initialized: GPIO={laser_control.gpio_available}, Pin={laser_control.pin_claimed}")

if not laser_control.gpio_available or not laser_control.pin_claimed:
    logging.error("LASER IS IN SIMULATION MODE!")
else:
    logging.info("LASER HAS GPIO CONTROL!")

@app.route('/api/laser/on', methods=['POST'])
def laser_on():
    if laser_control is None:
        return jsonify({'error': 'Laser control not available'}), 503
    try:
        result = laser_control.activate_laser()
        return jsonify({'message': 'Laser turned on', 'success': result}), 200
    except Exception as e:
        logging.error(f"Laser on error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/laser/off', methods=['POST'])
def laser_off():
    if laser_control is None:
        return jsonify({'error': 'Laser control not available'}), 503
    try:
        result = laser_control.deactivate_laser()
        return jsonify({'message': 'Laser turned off', 'success': result}), 200
    except Exception as e:
        logging.error(f"Laser off error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/laser/status', methods=['GET'])
def laser_status():
    if laser_control is None:
        return jsonify({'error': 'Laser control not available'}), 503
    try:
        status = laser_control.is_laser_active()
        in_simulation = not (laser_control.gpio_available and laser_control.pin_claimed)
        return jsonify({
            'laser_on': status,
            'simulation_mode': in_simulation,
            'gpio_available': laser_control.gpio_available,
            'pin_claimed': laser_control.pin_claimed
        }), 200
    except Exception as e:
        logging.error(f"Laser status error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        if laser_control:
            laser_control.cleanup()
            logging.info("Laser cleaned up")