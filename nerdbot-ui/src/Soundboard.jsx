import { useState, useEffect } from 'react';
import './Soundboard.css';

function Soundboard() {
  const [isOpen, setIsOpen] = useState(false);
  const [memeSounds, setMemeSounds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [urlInput, setUrlInput] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const memeSoundsEndpoint = 'http://10.0.1.204:5000/api/meme_sounds';
  const playMemeEndpoint = 'http://10.0.1.204:5000/api/meme_sound/';
  const addSoundEndpoint = 'http://10.0.1.204:5000/api/meme_sounds/add_from_url';
  const deleteSoundEndpoint = 'http://10.0.1.204:5000/api/meme_sounds/delete/';

  useEffect(() => {
    fetchMemeSounds();
  }, []);

  const fetchMemeSounds = async () => {
    setLoading(true);
    try {
      const response = await fetch(memeSoundsEndpoint);
      if (!response.ok) throw new Error('Failed to fetch meme sounds');
      const data = await response.json();
      setMemeSounds(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching meme sounds:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const playSound = async (index) => {
    try {
      const response = await fetch(playMemeEndpoint + index, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) throw new Error('Failed to play sound');
    } catch (err) {
      console.error('Error playing sound:', err);
    }
  };

  const addSoundFromUrl = async () => {
    if (!urlInput.trim()) {
      setError('Please enter a URL');
      return;
    }

    setIsAdding(true);
    setError(null);

    try {
      const response = await fetch(addSoundEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: urlInput.trim() }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to add sound');
      }

      // Clear the input
      setUrlInput('');
      
      // Refresh the sound list
      await fetchMemeSounds();
      
      setError(`Success: ${data.message}`);
      
      // Clear success message after 3 seconds
      setTimeout(() => setError(null), 3000);

    } catch (err) {
      console.error('Error adding sound:', err);
      setError(err.message);
    } finally {
      setIsAdding(false);
    }
  };

  const confirmDelete = (index, soundName) => {
    setDeleteConfirm({ index, soundName });
  };

  const cancelDelete = () => {
    setDeleteConfirm(null);
  };

  const deleteSound = async () => {
    if (!deleteConfirm) return;

    setIsDeleting(true);
    setError(null);

    try {
      const response = await fetch(deleteSoundEndpoint + deleteConfirm.index, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to delete sound');
      }

      // Close confirmation modal
      setDeleteConfirm(null);
      
      // Refresh the sound list
      await fetchMemeSounds();
      
      setError(`Success: ${data.message}`);
      
      // Clear success message after 3 seconds
      setTimeout(() => setError(null), 3000);

    } catch (err) {
      console.error('Error deleting sound:', err);
      setError(err.message);
    } finally {
      setIsDeleting(false);
    }
  };

  const formatSoundName = (filename) => {
    // Remove file extension and replace underscores/hyphens with spaces
    return filename
      .replace(/\.(mp3|wav|ogg|m4a)$/i, '')
      .replace(/[_-]/g, ' ')
      .replace(/\b\w/g, char => char.toUpperCase());
  };

  return (
    <>
      {/* Toggle Button */}
      <button 
        className="soundboard-toggle"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle soundboard"
      >
        {isOpen ? '🔇' : '🔊'}
      </button>

      {/* Soundboard Panel */}
      <div className={`soundboard-panel ${isOpen ? 'open' : ''}`}>
        <div className="soundboard-header">
          <h2>Soundboard</h2>
          <button 
            className="soundboard-close"
            onClick={() => setIsOpen(false)}
            aria-label="Close soundboard"
          >
            ✕
          </button>
        </div>

        <div className="soundboard-content">
          {loading && <div className="soundboard-loading">Loading sounds...</div>}
          {error && <div className={error.startsWith('Success:') ? 'soundboard-success' : 'soundboard-error'}>{error}</div>}
          
          {/* Add Sound from URL Section */}
          <div className="soundboard-add-section">
            <h3>Add New Sound</h3>
            <div className="soundboard-url-input">
              <input
                type="url"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="Paste sound URL (e.g., https://example.com/sound.mp3)"
                disabled={isAdding}
                onKeyDown={(e) => e.key === 'Enter' && addSoundFromUrl()}
              />
              <button
                onClick={addSoundFromUrl}
                disabled={isAdding || !urlInput.trim()}
                className="soundboard-add-button"
              >
                {isAdding ? '📥 Adding...' : '➕ Add'}
              </button>
            </div>
          </div>
          
          {!loading && (
            <div className="soundboard-grid">
              {memeSounds.map((sound, index) => (
                <div key={index} className="sound-item">
                  <button
                    className="sound-button"
                    onClick={() => playSound(index)}
                    title={sound}
                  >
                    <span className="sound-icon">🎵</span>
                    <span className="sound-name">{formatSoundName(sound)}</span>
                  </button>
                  <button
                    className="sound-delete-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      confirmDelete(index, formatSoundName(sound));
                    }}
                    title={`Delete ${formatSoundName(sound)}`}
                  >
                    🗑️
                  </button>
                </div>
              ))}
            </div>
          )}

          <button 
            className="soundboard-refresh"
            onClick={fetchMemeSounds}
            disabled={loading}
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div className="soundboard-modal-overlay">
          <div className="soundboard-modal">
            <h3>Confirm Delete</h3>
            <p>Are you sure you want to delete &ldquo;{deleteConfirm.soundName}&rdquo;?</p>
            <p className="soundboard-modal-warning">This action cannot be undone.</p>
            <div className="soundboard-modal-buttons">
              <button
                className="soundboard-modal-cancel"
                onClick={cancelDelete}
                disabled={isDeleting}
              >
                Cancel
              </button>
              <button
                className="soundboard-modal-delete"
                onClick={deleteSound}
                disabled={isDeleting}
              >
                {isDeleting ? '🗑️ Deleting...' : '🗑️ Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Overlay for mobile */}
      {isOpen && (
        <div 
          className="soundboard-overlay"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}

export default Soundboard;