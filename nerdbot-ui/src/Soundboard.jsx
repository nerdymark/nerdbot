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
  const [regeneratingIds, setRegeneratingIds] = useState(new Set());
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState('');
  const [reconvertingIds, setReconvertingIds] = useState(new Set());

  const memeSoundsEndpoint = 'http://10.0.1.204:5000/api/meme_sounds';
  const playMemeEndpoint = 'http://10.0.1.204:5000/api/meme_sound/';
  const addSoundEndpoint = 'http://10.0.1.204:5000/api/meme_sounds/add_from_url';
  const deleteSoundEndpoint = 'http://10.0.1.204:5000/api/meme_sounds/delete/';
  const regenerateThumbnailEndpoint = 'http://10.0.1.204:5000/api/meme_sounds/regenerate_thumbnail/';
  const renameSoundEndpoint = 'http://10.0.1.204:5000/api/meme_sounds/rename/';
  const reconvertSoundEndpoint = 'http://10.0.1.204:5000/api/meme_sounds/reconvert/';

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

  const playSound = async (sound) => {
    try {
      // Use the id property if it's an object, otherwise use the index
      const soundId = typeof sound === 'object' && sound.id !== undefined ? sound.id : sound;
      const response = await fetch(playMemeEndpoint + soundId, {
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

      // Display success or warning
      if (data.warning) {
        setError(`Success (with warning): ${data.message}. ${data.warning}`);
        setTimeout(() => setError(null), 7000); // Longer timeout for warnings
      } else {
        setError(`Success: ${data.message}`);
        setTimeout(() => setError(null), 3000);
      }

    } catch (err) {
      console.error('Error adding sound:', err);
      setError(err.message);
    } finally {
      setIsAdding(false);
    }
  };

  const confirmDelete = (sound, soundName) => {
    const soundId = typeof sound === 'object' && sound.id !== undefined ? sound.id : sound;
    setDeleteConfirm({ index: soundId, soundName });
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

  const startEdit = (sound, soundName) => {
    const soundId = typeof sound === 'object' && sound.id !== undefined ? sound.id : sound;
    setEditingId(soundId);
    setEditName(soundName.replace(/\.(mp3|wav|ogg|m4a)$/i, ''));
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditName('');
  };

  const saveRename = async (sound) => {
    const soundId = typeof sound === 'object' && sound.id !== undefined ? sound.id : sound;
    
    if (!editName.trim()) {
      setError('Name cannot be empty');
      return;
    }

    // Add extension if missing
    let newName = editName.trim();
    if (!newName.match(/\.(mp3|wav|ogg|m4a)$/i)) {
      // Get original extension
      const originalName = typeof sound === 'object' && sound.filename ? sound.filename : sound;
      const ext = originalName.match(/\.(mp3|wav|ogg|m4a)$/i);
      newName += ext ? ext[0] : '.mp3';
    }

    setError(null);

    try {
      const response = await fetch(renameSoundEndpoint + soundId, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ new_name: newName }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to rename sound');
      }

      // Close edit mode
      setEditingId(null);
      setEditName('');
      
      // Refresh the sound list
      await fetchMemeSounds();
      
      setError(`Success: ${data.message}`);
      
      // Clear success message after 3 seconds
      setTimeout(() => setError(null), 3000);

    } catch (err) {
      console.error('Error renaming sound:', err);
      setError(err.message);
    }
  };

  const reconvertSound = async (sound) => {
    const soundId = typeof sound === 'object' && sound.id !== undefined ? sound.id : sound;
    
    // Add to reconverting set
    setReconvertingIds(prev => new Set([...prev, soundId]));
    setError(null);

    try {
      const response = await fetch(reconvertSoundEndpoint + soundId, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to reconvert sound');
      }
      
      // Refresh to get updated sound
      await fetchMemeSounds();
      
      setError(`Success: Sound normalized and reconverted`);
      
      // Clear success message after 3 seconds
      setTimeout(() => setError(null), 3000);

    } catch (err) {
      console.error('Error reconverting sound:', err);
      setError(err.message);
    } finally {
      // Remove from reconverting set
      setReconvertingIds(prev => {
        const newSet = new Set(prev);
        newSet.delete(soundId);
        return newSet;
      });
    }
  };

  const regenerateThumbnail = async (sound) => {
    const soundId = typeof sound === 'object' && sound.id !== undefined ? sound.id : sound;
    
    // Add to regenerating set
    setRegeneratingIds(prev => new Set([...prev, soundId]));
    setError(null);

    try {
      const response = await fetch(regenerateThumbnailEndpoint + soundId, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to regenerate thumbnail');
      }

      // Update the thumbnail URL with cache-busting timestamp
      if (data.thumbnail_url) {
        // Force refresh the thumbnail by updating the sound list
        setMemeSounds(prevSounds =>
          prevSounds.map(s => {
            const sid = typeof s === 'object' && s.id !== undefined ? s.id : prevSounds.indexOf(s);
            if (sid === soundId) {
              return {
                ...s,
                thumbnail_url: data.thumbnail_url // Use the new URL with timestamp
              };
            }
            return s;
          })
        );
      }

      // Check for warnings (e.g., API quota issues)
      if (data.warning) {
        setError(`Warning: ${data.warning}`);
      } else {
        setError(`Success: Thumbnail regenerated for ${formatSoundName(sound)}`);
      }

      // Clear message after 5 seconds (longer for warnings)
      setTimeout(() => setError(null), 5000);

    } catch (err) {
      console.error('Error regenerating thumbnail:', err);
      setError(err.message);
    } finally {
      // Remove from regenerating set
      setRegeneratingIds(prev => {
        const newSet = new Set(prev);
        newSet.delete(soundId);
        return newSet;
      });
    }
  };

  const formatSoundName = (sound) => {
    // Handle object format from API
    if (typeof sound === 'object' && sound !== null) {
      // Use the name property if available
      if (sound.name) {
        return sound.name;
      }
      // Fall back to filename if name is not available
      if (sound.filename) {
        return sound.filename
          .replace(/\.(mp3|wav|ogg|m4a)$/i, '')
          .replace(/[_-]/g, ' ')
          .replace(/\b\w/g, char => char.toUpperCase());
      }
    }
    // Handle string format (backwards compatibility)
    if (typeof sound === 'string') {
      return sound
        .replace(/\.(mp3|wav|ogg|m4a)$/i, '')
        .replace(/[_-]/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase());
    }
    // Default fallback
    return 'Unknown Sound';
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
      <div 
        className={`soundboard-panel${isOpen ? ' open' : ''}`}
        style={{
          position: 'fixed',
          right: isOpen ? '0px' : '-450px',
          top: '0px',
          zIndex: 9998
        }}
      >
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
              {memeSounds.map((sound, index) => {
                const soundId = typeof sound === 'object' && sound.id !== undefined ? sound.id : index;
                const thumbnailUrl = typeof sound === 'object' && sound.thumbnail_url 
                  ? `http://10.0.1.204:5000${sound.thumbnail_url}`
                  : `http://10.0.1.204:5000/api/meme_sounds/thumbnail/${soundId}`;
                
                return (
                  <div key={soundId} className="sound-item">
                    <button
                      className="sound-button"
                      onClick={() => playSound(sound)}
                      title={formatSoundName(sound)}
                    >
                      <img 
                        src={thumbnailUrl}
                        alt={formatSoundName(sound)}
                        className="sound-thumbnail"
                        onError={(e) => {
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'block';
                        }}
                      />
                      <span className="sound-icon" style={{display: 'none'}}>🎵</span>
                    </button>
                    {editingId === soundId ? (
                      <div className="sound-edit-form">
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') saveRename(sound);
                            if (e.key === 'Escape') cancelEdit();
                          }}
                          className="sound-edit-input"
                          autoFocus
                        />
                        <div className="sound-edit-actions">
                          <button
                            className="sound-save-button"
                            onClick={() => saveRename(sound)}
                            title="Save"
                          >
                            ✓
                          </button>
                          <button
                            className="sound-cancel-button"
                            onClick={cancelEdit}
                            title="Cancel"
                          >
                            ✗
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <span className="sound-name">{formatSoundName(sound)}</span>
                        <div className="sound-actions">
                          <button
                            className="sound-action-button sound-edit-button"
                            onClick={(e) => {
                              e.stopPropagation();
                              startEdit(sound, formatSoundName(sound));
                            }}
                            title="Rename"
                          >
                            ✏️
                          </button>
                          <button
                            className="sound-action-button sound-normalize-button"
                            onClick={(e) => {
                              e.stopPropagation();
                              reconvertSound(sound);
                            }}
                            disabled={reconvertingIds.has(soundId)}
                            title="Normalize volume to 95%"
                          >
                            {reconvertingIds.has(soundId) ? '⏳' : '🔊'}
                          </button>
                          <button
                            className="sound-action-button sound-regenerate-button"
                            onClick={(e) => {
                              e.stopPropagation();
                              regenerateThumbnail(sound);
                            }}
                            disabled={regeneratingIds.has(soundId)}
                            title="Regenerate thumbnail"
                          >
                            {regeneratingIds.has(soundId) ? '⏳' : '🖼️'}
                          </button>
                          <button
                            className="sound-action-button sound-delete-button"
                            onClick={(e) => {
                              e.stopPropagation();
                              confirmDelete(sound, formatSoundName(sound));
                            }}
                            title="Delete"
                          >
                            🗑️
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
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