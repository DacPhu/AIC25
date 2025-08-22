/**
 * Frontend Service Discovery and Load Balancing
 * Automatically discovers and manages connections to multiple backend instances
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

export interface BackendService {
  service_id: string;
  service_type: string;
  host: string;
  port: number;
  version: string;
  status: string;
  capabilities: string[];
  metadata: Record<string, any>;
  last_heartbeat: number;
  device_id: string;
  device_name: string;
  base_url: string;
  response_time?: number;
  success_rate?: number;
  last_used?: number;
}

export interface DiscoveryConfig {
  // Registry endpoints to check for service discovery
  registryEndpoints: string[];
  // Default ports to scan on local network
  defaultPorts: number[];
  // Discovery interval in milliseconds
  discoveryInterval: number;
  // Health check interval in milliseconds
  healthCheckInterval: number;
  // Timeout for discovery requests in milliseconds
  discoveryTimeout: number;
  // Maximum number of backends to track
  maxBackends: number;
  // Load balancing strategy
  loadBalancingStrategy: 'round_robin' | 'random' | 'response_time' | 'health_weighted';
}

export class BackendDiscovery {
  private config: DiscoveryConfig;
  private availableBackends: Map<string, BackendService> = new Map();
  private currentBackendIndex = 0;
  private discoveryInterval?: NodeJS.Timeout;
  private healthCheckInterval?: NodeJS.Timeout;
  private isDiscovering = false;
  private eventListeners: Map<string, Function[]> = new Map();

  constructor(config: Partial<DiscoveryConfig> = {}) {
    this.config = {
      registryEndpoints: ['http://127.0.0.1:5000'],
      defaultPorts: [5000, 5001, 5002, 5003, 5004, 5005],
      discoveryInterval: 30000, // 30 seconds
      healthCheckInterval: 10000, // 10 seconds
      discoveryTimeout: 5000, // 5 seconds
      maxBackends: 10,
      loadBalancingStrategy: 'health_weighted',
      ...config
    };
  }

  /**
   * Start the discovery process
   */
  async start(): Promise<void> {
    console.log('Starting backend discovery...');
    
    // Initial discovery
    await this.discoverBackends();
    
    // Set up periodic discovery
    this.discoveryInterval = setInterval(() => {
      this.discoverBackends();
    }, this.config.discoveryInterval);
    
    // Set up periodic health checks
    this.healthCheckInterval = setInterval(() => {
      this.performHealthChecks();
    }, this.config.healthCheckInterval);
    
    this.emit('discovery:started');
  }

  /**
   * Stop the discovery process
   */
  stop(): void {
    console.log('Stopping backend discovery...');
    
    if (this.discoveryInterval) {
      clearInterval(this.discoveryInterval);
      this.discoveryInterval = undefined;
    }
    
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = undefined;
    }
    
    this.emit('discovery:stopped');
  }

  /**
   * Discover available backend services
   */
  private async discoverBackends(): Promise<void> {
    if (this.isDiscovering) return;
    
    this.isDiscovering = true;
    
    try {
      // Method 1: Query service registries
      await this.discoverFromRegistries();
      
      // Method 2: Scan common ports on local network
      await this.discoverByPortScanning();
      
      // Method 3: Check predefined endpoints
      await this.discoverPredefinedEndpoints();
      
      console.log(`Discovered ${this.availableBackends.size} backends`);
      this.emit('backends:updated', Array.from(this.availableBackends.values()));
      
    } catch (error) {
      console.error('Error during backend discovery:', error);
      this.emit('discovery:error', error);
    } finally {
      this.isDiscovering = false;
    }
  }

  /**
   * Query service registries for backend services
   */
  private async discoverFromRegistries(): Promise<void> {
    const discoveryPromises = this.config.registryEndpoints.map(async (endpoint) => {
      try {
        const response = await axios.get(`${endpoint}/api/v1/registry/services`, {
          params: { service_type: 'backend' },
          timeout: this.config.discoveryTimeout
        });
        
        if (response.data.success && response.data.services) {
          response.data.services.forEach((service: any) => {
            this.addBackend(this.createBackendService(service));
          });
        }
      } catch (error) {
        console.debug(`Failed to discover from registry ${endpoint}:`, error);
      }
    });
    
    await Promise.allSettled(discoveryPromises);
  }

  /**
   * Scan common ports on local network for backend services
   */
  private async discoverByPortScanning(): Promise<void> {
    const localIPs = await this.getLocalNetworkIPs();
    const scanPromises: Promise<void>[] = [];
    
    for (const ip of localIPs) {
      for (const port of this.config.defaultPorts) {
        scanPromises.push(this.checkEndpoint(`http://${ip}:${port}`));
      }
    }
    
    // Limit concurrent scans to avoid overwhelming the network
    const chunks = this.chunkArray(scanPromises, 10);
    for (const chunk of chunks) {
      await Promise.allSettled(chunk);
    }
  }

  /**
   * Check predefined endpoints
   */
  private async discoverPredefinedEndpoints(): Promise<void> {
    const predefinedEndpoints = [
      'http://localhost:5000',
      'http://127.0.0.1:5000',
      'http://0.0.0.0:5000'
    ];
    
    const checkPromises = predefinedEndpoints.map(endpoint => 
      this.checkEndpoint(endpoint)
    );
    
    await Promise.allSettled(checkPromises);
  }

  /**
   * Check if an endpoint is a valid backend service
   */
  private async checkEndpoint(baseUrl: string): Promise<void> {
    try {
      const response = await axios.get(`${baseUrl}/api/v1/system/health`, {
        timeout: this.config.discoveryTimeout
      });
      
      if (response.status === 200 && response.data.success) {
        // This is a valid backend, create service entry
        const url = new URL(baseUrl);
        const service: BackendService = {
          service_id: `discovered_${url.host}_${url.port}`,
          service_type: 'backend',
          host: url.hostname,
          port: parseInt(url.port) || 80,
          version: response.data.version || '2.0.0',
          status: 'healthy',
          capabilities: ['search', 'frames', 'videos', 'system'], // Assume full capabilities
          metadata: response.data,
          last_heartbeat: Date.now(),
          device_id: response.data.device_id || 'unknown',
          device_name: response.data.device_name || url.hostname,
          base_url: baseUrl
        };
        
        this.addBackend(service);
      }
    } catch (error) {
      // Endpoint is not a valid backend or not reachable
      console.debug(`Endpoint ${baseUrl} is not a valid backend`);
    }
  }

  /**
   * Perform health checks on all known backends
   */
  private async performHealthChecks(): Promise<void> {
    const healthCheckPromises = Array.from(this.availableBackends.values()).map(
      backend => this.checkBackendHealth(backend)
    );
    
    await Promise.allSettled(healthCheckPromises);
  }

  /**
   * Check health of a specific backend
   */
  private async checkBackendHealth(backend: BackendService): Promise<void> {
    const startTime = Date.now();
    
    try {
      const response = await axios.get(`${backend.base_url}/api/v1/system/health`, {
        timeout: this.config.discoveryTimeout
      });
      
      const responseTime = Date.now() - startTime;
      
      if (response.status === 200) {
        // Update backend health metrics
        backend.status = 'healthy';
        backend.response_time = responseTime;
        backend.success_rate = Math.min(1.0, (backend.success_rate || 0.8) + 0.1);
        backend.last_heartbeat = Date.now();
        
        this.availableBackends.set(backend.service_id, backend);
      } else {
        this.markBackendUnhealthy(backend);
      }
    } catch (error) {
      this.markBackendUnhealthy(backend);
    }
  }

  /**
   * Mark a backend as unhealthy
   */
  private markBackendUnhealthy(backend: BackendService): void {
    backend.status = 'unhealthy';
    backend.success_rate = Math.max(0.0, (backend.success_rate || 0.8) - 0.2);
    
    // Remove backend if it's been unhealthy for too long
    const timeSinceLastHeartbeat = Date.now() - backend.last_heartbeat;
    if (timeSinceLastHeartbeat > 60000) { // 1 minute
      this.availableBackends.delete(backend.service_id);
      console.log(`Removed unhealthy backend: ${backend.base_url}`);
      this.emit('backend:removed', backend);
    } else {
      this.availableBackends.set(backend.service_id, backend);
    }
  }

  /**
   * Add a backend to the available backends list
   */
  private addBackend(backend: BackendService): void {
    // Check if we already have this backend
    const existing = this.availableBackends.get(backend.service_id);
    if (existing && existing.base_url === backend.base_url) {
      // Update existing backend
      this.availableBackends.set(backend.service_id, {
        ...existing,
        ...backend,
        last_heartbeat: Date.now()
      });
    } else {
      // Add new backend
      if (this.availableBackends.size < this.config.maxBackends) {
        this.availableBackends.set(backend.service_id, backend);
        console.log(`Added new backend: ${backend.base_url}`);
        this.emit('backend:added', backend);
      }
    }
  }

  /**
   * Create a BackendService from service registry data
   */
  private createBackendService(serviceData: any): BackendService {
    return {
      service_id: serviceData.service_id,
      service_type: serviceData.service_type,
      host: serviceData.host,
      port: serviceData.port,
      version: serviceData.version,
      status: serviceData.status,
      capabilities: serviceData.capabilities || [],
      metadata: serviceData.metadata || {},
      last_heartbeat: serviceData.last_heartbeat,
      device_id: serviceData.device_id,
      device_name: serviceData.device_name,
      base_url: `http://${serviceData.host}:${serviceData.port}`
    };
  }

  /**
   * Get local network IP addresses to scan
   */
  private async getLocalNetworkIPs(): Promise<string[]> {
    // In a browser environment, we can't directly get network interfaces
    // So we'll use some common local network ranges
    const commonRanges = [
      '192.168.1', '192.168.0', '10.0.0', '172.16.0'
    ];
    
    const ips: string[] = [];
    
    // Add localhost and common local IPs
    ips.push('127.0.0.1', 'localhost');
    
    // For browser environment, we're limited in what we can scan
    // In a Node.js environment, you could use the 'os' module to get actual network interfaces
    
    return ips;
  }

  /**
   * Select the best backend based on the configured strategy
   */
  selectBestBackend(capabilities?: string[]): BackendService | null {
    const healthyBackends = Array.from(this.availableBackends.values())
      .filter(backend => backend.status === 'healthy');
    
    if (healthyBackends.length === 0) {
      return null;
    }
    
    // Filter by capabilities if specified
    let candidateBackends = healthyBackends;
    if (capabilities && capabilities.length > 0) {
      candidateBackends = healthyBackends.filter(backend =>
        capabilities.every(cap => backend.capabilities.includes(cap))
      );
    }
    
    if (candidateBackends.length === 0) {
      return null;
    }
    
    // Apply load balancing strategy
    switch (this.config.loadBalancingStrategy) {
      case 'round_robin':
        return this.selectRoundRobin(candidateBackends);
      case 'random':
        return this.selectRandom(candidateBackends);
      case 'response_time':
        return this.selectByResponseTime(candidateBackends);
      case 'health_weighted':
        return this.selectHealthWeighted(candidateBackends);
      default:
        return this.selectRandom(candidateBackends);
    }
  }

  /**
   * Round-robin selection
   */
  private selectRoundRobin(backends: BackendService[]): BackendService {
    const selected = backends[this.currentBackendIndex % backends.length];
    this.currentBackendIndex++;
    selected.last_used = Date.now();
    return selected;
  }

  /**
   * Random selection
   */
  private selectRandom(backends: BackendService[]): BackendService {
    const selected = backends[Math.floor(Math.random() * backends.length)];
    selected.last_used = Date.now();
    return selected;
  }

  /**
   * Select by response time (fastest first)
   */
  private selectByResponseTime(backends: BackendService[]): BackendService {
    const sorted = backends.sort((a, b) => (a.response_time || 1000) - (b.response_time || 1000));
    const selected = sorted[0];
    selected.last_used = Date.now();
    return selected;
  }

  /**
   * Health-weighted selection
   */
  private selectHealthWeighted(backends: BackendService[]): BackendService {
    const weights = backends.map(backend => {
      const responseScore = 1000 / Math.max(backend.response_time || 1000, 1);
      const successScore = (backend.success_rate || 0.8) * 100;
      return responseScore * 0.4 + successScore * 0.6;
    });
    
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    const random = Math.random() * totalWeight;
    
    let currentWeight = 0;
    for (let i = 0; i < backends.length; i++) {
      currentWeight += weights[i];
      if (currentWeight >= random) {
        const selected = backends[i];
        selected.last_used = Date.now();
        return selected;
      }
    }
    
    const selected = backends[backends.length - 1];
    selected.last_used = Date.now();
    return selected;
  }

  /**
   * Get all available backends
   */
  getAvailableBackends(): BackendService[] {
    return Array.from(this.availableBackends.values());
  }

  /**
   * Get healthy backends
   */
  getHealthyBackends(): BackendService[] {
    return Array.from(this.availableBackends.values())
      .filter(backend => backend.status === 'healthy');
  }

  /**
   * Get discovery statistics
   */
  getStats() {
    const backends = Array.from(this.availableBackends.values());
    const healthy = backends.filter(b => b.status === 'healthy');
    
    return {
      total_backends: backends.length,
      healthy_backends: healthy.length,
      unhealthy_backends: backends.length - healthy.length,
      strategy: this.config.loadBalancingStrategy,
      average_response_time: healthy.length > 0 
        ? healthy.reduce((sum, b) => sum + (b.response_time || 0), 0) / healthy.length 
        : 0,
      backends: backends.map(b => ({
        service_id: b.service_id,
        base_url: b.base_url,
        status: b.status,
        response_time: b.response_time,
        success_rate: b.success_rate,
        device_name: b.device_name,
        capabilities: b.capabilities
      }))
    };
  }

  /**
   * Event handling
   */
  on(event: string, callback: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event)!.push(callback);
  }

  private emit(event: string, data?: any): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(callback => callback(data));
    }
  }

  /**
   * Utility function to chunk an array
   */
  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }
}

// Global discovery instance
export const backendDiscovery = new BackendDiscovery();