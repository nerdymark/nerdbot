import { useState, useEffect } from 'react';
import './App.css'
import axios from 'axios';
import Soundboard from './Soundboard';

function App() {
  const [message, setMessage] = useState('');
  const [isVideoFeedActive, setIsVideoFeedActive] = useState(false);
  const [currentMode, setCurrentMode] = useState(null);
  const [headlightsOn, setHeadlightsOn] = useState(false);
  const frontCamera = 'http://10.0.1.204:5000/cam0'
  const rearCamera = 'http://10.0.1.204:5000/cam1'
  const visualDescriptionEndpoint = 'http://10.0.1.204:5000/api/visual_awareness'
  const audioStream = 'http://10.0.1.204:8000/Stream.mp3'
  // const memeSounds = 'http://10.0.1.204:5000/api/meme_sounds'
  const modeEndpoint = 'http://10.0.1.204:5000/api/mode'
  const [visualDescription, setVisualDescription] = useState(null);
  const [botVitals, setBotVitals] = useState(null);
  const [vitalsError, setVitalsError] = useState(null);

  // const cameras = [
  //   {
  //     name: 'Front Camera',
  //     url: frontCamera,
  //   },
  //   {
  //     name: 'Rear Camera',
  //     url: rearCamera,
  //   },
  // ]

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

  const handleHeadlightToggle = async () => {
    try {
      const response = await fetch('http://10.0.1.204:5000/api/headlights/toggle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setHeadlightsOn(data.headlights_on);
        console.log('Headlights toggled:', data.headlights_on);
      } else {
        console.error('Failed to toggle headlights');
      }
    } catch (error) {
      console.error('Error toggling headlights:', error);
    }
  }
  
  const handleModeChange = async (mode) => {
    console.log('handleModeChange called with mode:', mode);
    
    try {
      const response = await fetch(modeEndpoint + '/' + mode, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setCurrentMode(data.mode);
        console.log('Mode changed to:', data.mode);
      } else {
        console.error('Failed to change mode');
      }
    } catch (error) {
      console.error('Error changing mode:', error);
    }
  }
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  }

  const handleSendMessage = async (messageObj) => {
    console.log('handleSendMessage called');
    console.log('Message:', messageObj.content);
  
    if (messageObj.content.trim() === '') {
      console.log('Message is empty, returning');
      return;
    }
  
    try {
      console.log('Sending POST request to http://10.0.1.204:5000/api/tts/' + encodeURIComponent(messageObj.content));
      const response = await fetch('http://10.0.1.204:5000/api/tts/' + encodeURIComponent(messageObj.content), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: messageObj.content }),
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

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isVideoFeedActive) return;

      switch (e.key) {
        case 'w':
          handleMotorControl('forward');
          break;
        case 'a':
          handleMotorControl('strafe_left');
          break;
        case 's':
          handleMotorControl('backward');
          break;
        case 'd':
          handleMotorControl('strafe_right');
          break;
        case 'q':
          handleMotorControl('left');
          break;
        case 'e':
          handleMotorControl('right');
          break;
        default:
          break;
      }
    };

    const handleKeyUp = () => {
      if (!isVideoFeedActive) return;
      handleMotorControl('stop');
    };

    if (isVideoFeedActive) {
      window.addEventListener('keydown', handleKeyDown);
      window.addEventListener('keyup', handleKeyUp);
    } else {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [isVideoFeedActive]);

  useEffect(() => {
    const fetchCurrentMode = async () => {
      try {
        const response = await fetch(modeEndpoint);
        if (response.ok) {
          const data = await response.json();
          setCurrentMode(data.mode);
        }
      } catch (error) {
        console.error('Error fetching current mode:', error);
      }
    };

    // Initial fetch
    fetchCurrentMode();
    
    // Poll every 5 seconds to keep mode in sync
    const intervalId = setInterval(fetchCurrentMode, 5000);
    
    // Cleanup
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    const fetchHeadlightStatus = async () => {
      try {
        const response = await fetch('http://10.0.1.204:5000/api/headlights/status');
        if (response.ok) {
          const data = await response.json();
          setHeadlightsOn(data.headlights_on);
        }
      } catch (error) {
        console.error('Error fetching headlight status:', error);
      }
    };

    fetchHeadlightStatus();
  }, []);

  useEffect(() => {
    const fetchVisualDescription = async () => {
      try {
        const response = await axios.get(visualDescriptionEndpoint);
        setVisualDescription(response.data[0]);
      } catch (error) {
        console.error('Error fetching visual description:', error);
      }
    };

    fetchVisualDescription();
  }, []);

  useEffect(() => {
    const ws = new WebSocket('ws://your-backend-url/vitals');
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received vitals:', data);
        setBotVitals(data);
      } catch (err) {
        console.error('Error parsing vitals:', err);
        setVitalsError(err.message);
      }
    };
  
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setVitalsError('Failed to connect to robot');
    };
  
    return () => ws.close();
  }, []);

  useEffect(() => {
    const fetchVitals = async () => {
      try {
        const response = await fetch('http://10.0.1.204:5000/api/vitals');
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        console.log('Vitals data:', data);
        setBotVitals(data);
        setVitalsError(null);
      } catch (error) {
        console.error('Error fetching vitals:', error);
        setVitalsError(error.message);
      }
    };

    // Initial fetch
    fetchVitals();

    // Setup polling interval
    const intervalId = setInterval(fetchVitals, 1000);

    // Cleanup
    return () => clearInterval(intervalId);
  }, []);

  return (
    <>
      <h1>nerdbot</h1>
      <Soundboard />
      <div className="card">
        <div className="cameras">
          <div
            className="video-container"
            onMouseEnter={() => setIsVideoFeedActive(true)}
            onMouseLeave={() => setIsVideoFeedActive(false)}
          >
            <div className="video-feed">
              <img src={frontCamera} alt="Front Camera" />
              <div className="pip">
                <img src={rearCamera} alt="Rear Camera" />
              </div>
            </div>
          </div>
        </div>
        <div className="container">
            <div className="chat">
                <div className="chat-box">
                    <div className="chat-input">
                    <input
                        type="text"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyDown={handleKeyPress}
                        placeholder="Enter your message - It will be spoken by the robot"
                        className="message-input"
                    />
                    <button onClick={() => handleSendMessage({ type: "tts", content: message })}>Send</button>
                    </div>
                </div>
            </div>
        </div>

        <div className="controls">
        <div className="mode-switcher">
          {currentMode === null ? (
            <div>Loading mode...</div>
          ) : (
            <>
              <button 
                className={`mode-button ${currentMode === 'manual' ? 'active' : ''}`}
                onClick={() => handleModeChange('manual')}
              >
                Manual
              </button>
              <button 
                className={`mode-button ${currentMode === 'idle' ? 'active' : ''}`}
                onClick={() => handleModeChange('idle')}
              >
                Idle
              </button>
              <button 
                className={`mode-button ${currentMode === 'detect_and_follow' ? 'active' : ''}`}
                onClick={() => handleModeChange('detect_and_follow')}
              >
                Pan/Tilt Follow
              </button>
              <button 
                className={`mode-button ${currentMode === 'detect_and_follow_wheels' ? 'active' : ''}`}
                onClick={() => handleModeChange('detect_and_follow_wheels')}
              >
                Wheel Follow
              </button>
            </>
          )}
        </div>
        <em>mouse-over the video to enable keyboard controls</em>
          <div className="motor-grid-container">
            <button onClick={() => handleMotorControl('forward')}>Forward</button>
            <button onClick={() => handleMotorControl('left')}>Left</button>
            <button onClick={() => handleMotorControl('stop')}>Stop</button>
            <button onClick={() => handleMotorControl('right')}>Right</button>
            <button onClick={() => handleMotorControl('backward')}>Backward</button>
            <button onClick={() => handleMotorControl('strafe_left')}>Strafe Left</button>
            <button onClick={() => handleMotorControl('strafe_right')}>Strafe Right</button>
          </div>
          <div className="servo-grid-container">
            <button onClick={() => handlePanControl('left')}>Pan Left</button>
            <button onClick={() => handlePanControl('center')}>Center</button>
            <button onClick={() => handlePanControl('right')}>Pan Right</button>
            <button onClick={() => handleTiltControl('up')}>Tilt Up</button>
            <button onClick={() => handleTiltControl('down')}>Tilt Down</button>
          </div>
          <audio controls>
            <source src={audioStream} type="audio/x-wav;codec=pcm"/>
          </audio>
          <button onClick={handleRandomMemAudio}>Random Meme Sound</button>
          <button 
            onClick={handleHeadlightToggle}
            className={`headlight-button ${headlightsOn ? 'active' : ''}`}
            style={{
              backgroundColor: headlightsOn ? '#fff' : '#333',
              color: headlightsOn ? '#000' : '#fff',
              border: '2px solid #fff',
              padding: '10px 20px',
              borderRadius: '5px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            💡 Headlights {headlightsOn ? 'ON' : 'OFF'}
          </button>
        </div>

        {/* System Vitals Card */}
        {vitalsError ? (
          <div className="error-message">Error: {vitalsError}</div>
        ) : !botVitals ? (
          <div className="loading">Loading vitals...</div>
        ) : (
          <div className="vitals-card">
            <h3>System Vitals</h3>
            <div className="vital-row">
              <span>🔋 Battery</span>
              <div className="progress-bar">
                <div 
                  className="progress" 
                  style={{
                    width: `${botVitals.battery}%`, 
                    backgroundColor: botVitals.battery < 20 ? 'var(--error)' : 'var(--success)'
                  }}
                ></div>
              </div>
              <span>{botVitals.battery?.toFixed(1)}% ({botVitals.battery_voltage?.toFixed(2)}V)</span>
            </div>
            <div className="vital-row">
              <span>💻 CPU</span>
              <div className="progress-bar">
                <div 
                  className="progress" 
                  style={{
                    width: `${botVitals.cpu}%`,
                    backgroundColor: botVitals.cpu > 80 ? 'var(--error)' : 'var(--success)'
                  }}
                ></div>
              </div>
              <span>{botVitals.cpu}%</span>
            </div>
            <div className="vital-row">
              <span>🧠 Memory</span>
              <div className="progress-bar">
                <div 
                  className="progress" 
                  style={{
                    width: `${botVitals.memory}%`,
                    backgroundColor: botVitals.memory > 80 ? 'var(--error)' : 'var(--success)'
                  }}
                ></div>
              </div>
              <span>{botVitals.memory}%</span>
            </div>
            <div className="vital-row">
              <span>🌡️ Temp</span>
              <span>{botVitals.temperature}°C</span>
            </div>
          </div>
        )}

        {visualDescription ? (
            <div className="visual-description">
                <div>
                <h3>What&apos;s in front of me?</h3>
                <p>{visualDescription.front}</p>
                <button 
                    className="tts-button"
                    onClick={() => handleSendMessage({
                        type: "tts",
                        content: visualDescription.front
                    })}
                >
                    🔊 Speak
                </button>
                </div>
                <div>
                <h3>What&apos;s behind me?</h3>
                <p>{visualDescription.rear}</p>
                <button 
                    className="tts-button"
                    onClick={() => handleSendMessage({
                        type: "tts", 
                        content: visualDescription.rear
                    })}
                >
                    🔊 Speak
                </button>
                </div>
            </div>
            ) : (
            <div className="visual-description">
                <p>Loading...</p>
            </div>
            )}
      </div>
    </>
  )
}

export default App