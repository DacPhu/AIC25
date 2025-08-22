/**
 * Simple Backend Discovery for Manual Distributed Deployment
 * Each device runs FastAPI backend manually, frontend can connect to any backend
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

export interface SimpleBackend {
  name: string;
  host: string;
  port: number;
  base_url: string;
  status: 'healthy' | 'unhealthy' | 'checking';
  response_time?: number;
  last_check?: number;
}

export class SimpleBackendManager {
  private backends: Map<string, SimpleBackend> = new Map();
  private currentBackend: string | null = null;
  private healthCheckInterval?: NodeJS.Timeout;
  private healthCheckRunning = false;

  constructor() {
    // Load backends from environment or localStorage
    this.loadBackendsFromConfig();
  }

  /**
   * Load backends from configuration
   */
  private loadBackendsFromConfig() {
    // Check environment variables first
    const backendHosts = import.meta.env.VITE_BACKEND_HOSTS;
    if (backendHosts) {
      const hosts = backendHosts.split(',');
      hosts.forEach((host: string, index: number) => {
        const [hostname, port] = host.trim().split(':');
        this.addBackend(`backend_${index + 1}`, hostname, parseInt(port) || 5000);
      });
    }

    // Load from localStorage
    const savedBackends = localStorage.getItem('aic25_backends');
    if (savedBackends) {
      try {
        const backends = JSON.parse(savedBackends);
        backends.forEach((backend: any) => {
          this.addBackend(backend.name, backend.host, backend.port);
        });
      } catch (error) {
        console.error('Failed to load saved backends:', error);
      }
    }

    // Default backends if none configured
    if (this.backends.size === 0) {
      this.addBackend('local', '127.0.0.1', 5000);
      this.addBackend('localhost', 'localhost', 5000);
    }
  }

  /**
   * Add a backend manually
   */
  addBackend(name: string, host: string, port: number): void {
    const backend: SimpleBackend = {
      name,
      host,
      port,
      base_url: `http://${host}:${port}`,
      status: 'checking'
    };
    
    this.backends.set(name, backend);
    this.saveBackendsToStorage();
    
    // Check health immediately
    this.checkBackendHealth(name);
    
    console.log(`Added backend: ${name} (${backend.base_url})`);
  }

  /**
   * Remove a backend
   */
  removeBackend(name: string): void {
    if (this.backends.delete(name)) {
      this.saveBackendsToStorage();
      if (this.currentBackend === name) {
        this.currentBackend = null;
      }
      console.log(`Removed backend: ${name}`);
    }
  }

  /**
   * Save backends to localStorage
   */
  private saveBackendsToStorage(): void {
    const backends = Array.from(this.backends.values()).map(backend => ({
      name: backend.name,
      host: backend.host,
      port: backend.port
    }));
    localStorage.setItem('aic25_backends', JSON.stringify(backends));
  }

  /**
   * Get all backends
   */
  getBackends(): SimpleBackend[] {
    return Array.from(this.backends.values());
  }

  /**
   * Get healthy backends only
   */
  getHealthyBackends(): SimpleBackend[] {
    return Array.from(this.backends.values()).filter(b => b.status === 'healthy');
  }

  /**
   * Check health of a specific backend
   */
  async checkBackendHealth(name: string): Promise<void> {
    const backend = this.backends.get(name);
    if (!backend) return;

    backend.status = 'checking';
    const startTime = Date.now();

    try {
      const response = await axios.get(`${backend.base_url}/api/v1/system/health`, {
        timeout: 5000
      });

      if (response.status === 200 && response.data.success) {
        backend.status = 'healthy';
        backend.response_time = Date.now() - startTime;
        backend.last_check = Date.now();
      } else {
        backend.status = 'unhealthy';
      }
    } catch (error) {
      backend.status = 'unhealthy';
      backend.response_time = undefined;
    }

    this.backends.set(name, backend);
  }

  /**
   * Check health of all backends
   */
  async checkAllBackends(): Promise<void> {
    if (this.healthCheckRunning) return;
    
    this.healthCheckRunning = true;
    
    const checks = Array.from(this.backends.keys()).map(name => 
      this.checkBackendHealth(name)
    );
    
    await Promise.allSettled(checks);
    this.healthCheckRunning = false;
  }

  /**
   * Start periodic health checks
   */
  startHealthChecks(interval: number = 30000): void {
    this.stopHealthChecks();
    
    // Initial check
    this.checkAllBackends();
    
    // Periodic checks
    this.healthCheckInterval = setInterval(() => {
      this.checkAllBackends();
    }, interval);
    
    console.log(`Started health checks (${interval}ms interval)`);
  }

  /**
   * Stop health checks
   */
  stopHealthChecks(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = undefined;
      console.log('Stopped health checks');
    }
  }

  /**
   * Select the best available backend
   */
  selectBestBackend(): SimpleBackend | null {
    const healthyBackends = this.getHealthyBackends();
    
    if (healthyBackends.length === 0) {
      return null;
    }

    // If we have a current backend and it's healthy, prefer it
    if (this.currentBackend) {
      const current = this.backends.get(this.currentBackend);
      if (current && current.status === 'healthy') {
        return current;
      }
    }

    // Select fastest backend
    const fastest = healthyBackends.reduce((best, current) => {
      if (!best.response_time) return current;
      if (!current.response_time) return best;
      return current.response_time < best.response_time ? current : best;
    });

    this.currentBackend = fastest.name;
    return fastest;
  }

  /**
   * Create axios client for a specific backend
   */
  createClient(backend: SimpleBackend): AxiosInstance {
    return axios.create({
      baseURL: `${backend.base_url}/api/v1`,
      timeout: 30000,
      headers: {
        'Accept-Encoding': 'gzip, deflate, br',
        'Content-Type': 'application/json'
      }
    });
  }

  /**
   * Execute request with automatic backend selection and failover
   */
  async executeRequest<T>(
    requestFn: (client: AxiosInstance, backend: SimpleBackend) => Promise<T>,
    maxRetries: number = 3
  ): Promise<T> {
    const healthyBackends = this.getHealthyBackends();
    
    if (healthyBackends.length === 0) {
      throw new Error('No healthy backends available');
    }

    let lastError: any;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      const backend = this.selectBestBackend();
      
      if (!backend) {
        throw new Error('No backends available');
      }

      try {
        const client = this.createClient(backend);
        const result = await requestFn(client, backend);
        
        // Success - update backend health
        backend.status = 'healthy';
        backend.last_check = Date.now();
        
        return result;
        
      } catch (error) {
        lastError = error;
        console.warn(`Request failed for backend ${backend.name}:`, error);
        
        // Mark backend as unhealthy
        backend.status = 'unhealthy';
        
        // Remove from current backend selection
        if (this.currentBackend === backend.name) {
          this.currentBackend = null;
        }
        
        // Wait before retry
        if (attempt < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
    }

    throw lastError || new Error('All backend requests failed');
  }

  /**
   * Get backend statistics
   */
  getStats() {
    const backends = Array.from(this.backends.values());
    const healthy = backends.filter(b => b.status === 'healthy');
    
    return {
      total_backends: backends.length,
      healthy_backends: healthy.length,
      unhealthy_backends: backends.length - healthy.length,
      current_backend: this.currentBackend,
      average_response_time: healthy.length > 0 
        ? healthy.reduce((sum, b) => sum + (b.response_time || 0), 0) / healthy.length 
        : 0,
      backends: backends.map(b => ({
        name: b.name,
        host: b.host,
        port: b.port,
        status: b.status,
        response_time: b.response_time,
        last_check: b.last_check
      }))
    };
  }

  /**
   * Discover backends on local network
   */
  async discoverLocalBackends(): Promise<void> {
    console.log('Discovering backends on local network...');
    
    // Common ports to check
    const ports = [5000, 5001, 5002, 5003, 5004, 5005];
    
    // Get local network range (simplified)
    const localIPs = this.generateLocalIPs();
    
    const discoveries: Promise<void>[] = [];
    
    for (const ip of localIPs) {
      for (const port of ports) {
        discoveries.push(this.tryDiscoverBackend(ip, port));
      }
    }
    
    // Limit concurrent requests
    const chunks = this.chunkArray(discoveries, 10);
    for (const chunk of chunks) {
      await Promise.allSettled(chunk);
    }
    
    console.log(`Discovery complete. Found ${this.getHealthyBackends().length} healthy backends.`);
  }

  /**
   * Try to discover a backend at specific IP:port
   */
  private async tryDiscoverBackend(ip: string, port: number): Promise<void> {
    try {
      const response = await axios.get(`http://${ip}:${port}/api/v1/system/health`, {
        timeout: 3000
      });
      
      if (response.status === 200 && response.data.success) {
        const backendName = `discovered_${ip}_${port}`;
        if (!this.backends.has(backendName)) {
          this.addBackend(backendName, ip, port);
          console.log(`🔍 Discovered backend: ${ip}:${port}`);
        }
      }
    } catch (error) {
      // Ignore discovery failures
    }
  }

  /**
   * Generate common local IP addresses to scan
   */
  private generateLocalIPs(): string[] {
    const ips = ['127.0.0.1', 'localhost'];
    
    // Add common local network ranges
    for (let i = 1; i < 10; i++) {
      ips.push(`192.168.1.${i}`);
      ips.push(`192.168.0.${i}`);
      ips.push(`10.0.0.${i}`);
    }
    
    return ips;
  }

  /**
   * Utility to chunk array
   */
  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }
}

// Global instance
export const simpleBackendManager = new SimpleBackendManager();