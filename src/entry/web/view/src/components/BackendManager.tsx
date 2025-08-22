/**
 * Backend Manager Component
 * Allows users to manually add/remove backends and see their status
 */

import React, { useState, useEffect } from 'react';
import { 
  getBackends, 
  getBackendStats, 
  addBackend, 
  removeBackend, 
  discoverBackends,
  checkBackendHealth 
} from '../services/distributed_search';

interface Backend {
  name: string;
  host: string;
  port: number;
  status: 'healthy' | 'unhealthy' | 'checking';
  response_time?: number;
  last_check?: number;
}

export const BackendManager: React.FC = () => {
  const [backends, setBackends] = useState<Backend[]>([]);
  const [stats, setStats] = useState<any>({});
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [newBackend, setNewBackend] = useState({
    name: '',
    host: '',
    port: 5000
  });

  // Refresh data
  const refreshData = () => {
    setBackends(getBackends());
    setStats(getBackendStats());
  };

  // Auto-refresh every 5 seconds
  useEffect(() => {
    refreshData();
    const interval = setInterval(refreshData, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleAddBackend = () => {
    if (newBackend.name && newBackend.host) {
      addBackend(newBackend.name, newBackend.host, newBackend.port);
      setNewBackend({ name: '', host: '', port: 5000 });
      refreshData();
    }
  };

  const handleRemoveBackend = (name: string) => {
    removeBackend(name);
    refreshData();
  };

  const handleDiscover = async () => {
    setIsDiscovering(true);
    try {
      await discoverBackends();
      refreshData();
    } finally {
      setIsDiscovering(false);
    }
  };

  const handleHealthCheck = async () => {
    await checkBackendHealth();
    refreshData();
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return '✅';
      case 'unhealthy':
        return '❌';
      case 'checking':
        return '🔄';
      default:
        return '❓';
    }
  };

  const formatResponseTime = (time?: number) => {
    if (!time) return 'N/A';
    return `${Math.round(time)}ms`;
  };

  const formatLastCheck = (timestamp?: number) => {
    if (!timestamp) return 'Never';
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  return (
    <div className="backend-manager" style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h2>🔧 Backend Manager</h2>
      
      {/* Statistics */}
      <div style={{ 
        background: '#f5f5f5', 
        padding: '15px', 
        borderRadius: '8px', 
        marginBottom: '20px' 
      }}>
        <h3>📊 Statistics</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
          <div>
            <strong>Total Backends:</strong> {stats.total_backends || 0}
          </div>
          <div>
            <strong>Healthy:</strong> {stats.healthy_backends || 0}
          </div>
          <div>
            <strong>Unhealthy:</strong> {stats.unhealthy_backends || 0}
          </div>
          <div>
            <strong>Avg Response:</strong> {formatResponseTime(stats.average_response_time)}
          </div>
          <div>
            <strong>Current Backend:</strong> {stats.current_backend || 'None'}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={handleDiscover}
          disabled={isDiscovering}
          style={{ 
            marginRight: '10px', 
            padding: '8px 16px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isDiscovering ? 'not-allowed' : 'pointer'
          }}
        >
          {isDiscovering ? '🔍 Discovering...' : '🔍 Auto-Discover'}
        </button>
        
        <button 
          onClick={handleHealthCheck}
          style={{ 
            padding: '8px 16px',
            backgroundColor: '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          🩺 Health Check
        </button>
      </div>

      {/* Add Backend Form */}
      <div style={{ 
        background: '#e9ecef', 
        padding: '15px', 
        borderRadius: '8px', 
        marginBottom: '20px' 
      }}>
        <h3>➕ Add Backend</h3>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            type="text"
            placeholder="Backend Name"
            value={newBackend.name}
            onChange={(e) => setNewBackend({ ...newBackend, name: e.target.value })}
            style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
          />
          <input
            type="text"
            placeholder="Host (IP/hostname)"
            value={newBackend.host}
            onChange={(e) => setNewBackend({ ...newBackend, host: e.target.value })}
            style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
          />
          <input
            type="number"
            placeholder="Port"
            value={newBackend.port}
            onChange={(e) => setNewBackend({ ...newBackend, port: parseInt(e.target.value) || 5000 })}
            style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc', width: '80px' }}
          />
          <button 
            onClick={handleAddBackend}
            style={{ 
              padding: '8px 16px',
              backgroundColor: '#17a2b8',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Add
          </button>
        </div>
        <div style={{ fontSize: '12px', color: '#666', marginTop: '5px' }}>
          Example: Name="Device2", Host="192.168.1.100", Port="5001"
        </div>
      </div>

      {/* Backend List */}
      <div>
        <h3>🖥️ Available Backends ({backends.length})</h3>
        {backends.length === 0 ? (
          <div style={{ 
            padding: '20px', 
            textAlign: 'center', 
            color: '#666',
            border: '2px dashed #ccc',
            borderRadius: '8px'
          }}>
            No backends configured. Add a backend above or click "Auto-Discover".
          </div>
        ) : (
          <div style={{ display: 'grid', gap: '10px' }}>
            {backends.map((backend) => (
              <div 
                key={backend.name}
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  padding: '12px',
                  border: `2px solid ${backend.status === 'healthy' ? '#28a745' : '#dc3545'}`,
                  borderRadius: '8px',
                  backgroundColor: backend.status === 'healthy' ? '#d4edda' : '#f8d7da'
                }}
              >
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '16px', fontWeight: 'bold' }}>
                    {getStatusIcon(backend.status)} {backend.name}
                  </div>
                  <div style={{ fontSize: '14px', color: '#666' }}>
                    {backend.host}:{backend.port}
                  </div>
                  <div style={{ fontSize: '12px', color: '#888' }}>
                    Response: {formatResponseTime(backend.response_time)} | 
                    Last Check: {formatLastCheck(backend.last_check)}
                  </div>
                </div>
                <button 
                  onClick={() => handleRemoveBackend(backend.name)}
                  style={{ 
                    padding: '6px 12px',
                    backgroundColor: '#dc3545',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '12px'
                  }}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Instructions */}
      <div style={{ 
        marginTop: '30px', 
        padding: '15px', 
        background: '#fff3cd', 
        border: '1px solid #ffeaa7',
        borderRadius: '8px',
        fontSize: '14px'
      }}>
        <h4>📝 Usage Instructions:</h4>
        <ol>
          <li><strong>Start backends manually</strong> on each device:
            <br /><code>python scripts/start_backend.py --host 0.0.0.0 --port 5000</code>
          </li>
          <li><strong>Add backends</strong> using the form above or click "Auto-Discover"</li>
          <li><strong>Frontend automatically</strong> load balances across healthy backends</li>
          <li><strong>Cross-device example:</strong>
            <ul>
              <li>Device 1: Run backend on port 5000</li>
              <li>Device 2: Run backend on port 5001</li>
              <li>Device 1 frontend can connect to Device 2 backend automatically</li>
            </ul>
          </li>
        </ol>
      </div>
    </div>
  );
};

export default BackendManager;