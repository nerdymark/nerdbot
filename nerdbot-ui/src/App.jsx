import React, { useState } from 'react';
import './App.css'

function App() {
  const [message, setMessage] = useState('');
  const frontCamera = 'http://10.0.1.204:5000/cam0'
  const rearCamera = 'http://10.0.1.204:5000/cam1'
  const audioStream = 'http://10.0.1.204:5000/audio'

  const cameras = [
    {
      name: 'Front Camera',
      url: frontCamera,
    },
    {
      name: 'Rear Camera',
      url: rearCamera,
    },
  ]

  const randomMemAudio = 'http://10.0.1.204:5000/api/meme_sound/random'
  const handleRandomMemAudio = () => {
    console.log('handleRandomMemAudio called');

    try {
      console.log('Sending POST request to', randomMemAudio);
      fetch(randomMemAudio, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    }
    catch (error) {
      console.error('Error:', error);
    }

  }

  const motorEndpoint = 'http://10.0.1.204:5000/api/motor/'
  const handleMotorControl = (direction) => {
    console.log('handleMotorControl called');
    console.log('Direction:', direction);

    try {
      console.log('Sending POST request to', motorEndpoint + direction);
      fetch(motorEndpoint + direction, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    }
    catch (error) {
      console.error('Error:', error);
    }

  }

  const panEndpoint = 'http://10.0.1.204:5000/api/pan/'
  const handlePanControl = (direction) => {
    console.log('handlePanControl called');
    console.log('Direction:', direction);

    try {
      console.log('Sending POST request to', panEndpoint + direction);
      fetch(panEndpoint + direction, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    }
    catch (error) {
      console.error('Error:', error);
    }

  }

  const tiltEndpoint = 'http://10.0.1.204:5000/api/tilt/'
  const handleTiltControl = (direction) => {
    console.log('handleTiltControl called');
    console.log('Direction:', direction);

    try {
      console.log('Sending POST request to', tiltEndpoint + direction);
      fetch(tiltEndpoint + direction, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    }
    catch (error) {
      console.error('Error:', error
      );
    }

  }
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  }

  const handleSendMessage = async () => {
    console.log('handleSendMessage called');
    console.log('Message:', message);
  
    if (message.trim() === '') {
      console.log('Message is empty, returning');
      return;
    }
  
    try {
      console.log('Sending POST request to http://10.0.1.204:5000/api/tts/' + encodeURIComponent(message));
      const response = await fetch('http://10.0.1.204:5000/api/tts/' + encodeURIComponent(message), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });
  
      console.log('Response status:', response.status);
  
      if (response.ok) {
        console.log('Message sent successfully');
        setMessage(''); // Clear the input field
      } else {
        console.error('Failed to send message');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  }
  return (
    <>
      <h1>nerdbot</h1>
      <div className="card">
        <div className="cameras">
          <div className="video-container">
            <div className="video-feed">
              <img src={frontCamera} alt="Front Camera" />
              <div className="pip">
                <img src={rearCamera} alt="Rear Camera" />
              </div>
            </div>
          </div>
        </div>
        <div className="chat">
          <div className="chat-box">
            <div className="chat-input">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Enter your message..."
                className="message-input"
              />
              <button onClick={handleSendMessage}>Send</button>
            </div>
          </div>
        </div>

        <div className="controls">
          <div className="motor-grid-container">
            <button class="disable-dbl-tap-zoom" onClick={() => handleMotorControl('forward')}>Forward</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handleMotorControl('left')}>Left</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handleMotorControl('stop')}>Stop</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handleMotorControl('right')}>Right</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handleMotorControl('backward')}>Backward</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handleMotorControl('strafe_left')}>Strafe Left</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handleMotorControl('strafe_right')}>Strafe Right</button>
          </div>
          <div className="servo-grid-container">
            <button class="disable-dbl-tap-zoom" onClick={() => handlePanControl('left')}>Pan Left</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handlePanControl('center')}>Center</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handlePanControl('right')}>Pan Right</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handleTiltControl('up')}>Tilt Up</button>
            <button class="disable-dbl-tap-zoom" onClick={() => handleTiltControl('down')}>Tilt Down</button>
          </div>
          <audio controls>
            <source src={audioStream} type="audio/x-wav;codec=pcm"/>
          </audio>
          <button class="disable-dbl-tap-zoom" onClick={handleRandomMemAudio}>Random Meme Sound</button>
        </div>
      </div>
    </>
  )
}

export default App
