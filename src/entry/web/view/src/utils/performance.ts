// Performance monitoring utilities

class PerformanceMonitor {
  constructor() {
    this.metrics = {};
    this.observers = [];
    this.init();
  }

  init() {
    this.measureCoreWebVitals();

    this.measureResourceTiming();
    
    // User interaction monitoring
    this.measureUserInteractions();
    
    // Memory usage monitoring (if available)
    this.measureMemoryUsage();
  }

  measureCoreWebVitals() {
    // First Contentful Paint (FCP)
    this.measureFCP();
    
    // Largest Contentful Paint (LCP)
    this.measureLCP();
    
    // First Input Delay (FID)
    this.measureFID();
    
    // Cumulative Layout Shift (CLS)
    this.measureCLS();
  }

  measureFCP() {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === 'first-contentful-paint') {
          this.metrics.fcp = entry.startTime;
          this.logMetric('FCP', entry.startTime);
        }
      }
    }).observe({ type: 'paint', buffered: true });
  }

  measureLCP() {
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.metrics.lcp = lastEntry.startTime;
      this.logMetric('LCP', lastEntry.startTime);
    }).observe({ type: 'largest-contentful-paint', buffered: true });
  }

  measureFID() {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.name === 'first-input') {
          const fid = entry.processingStart - entry.startTime;
          this.metrics.fid = fid;
          this.logMetric('FID', fid);
        }
      }
    }).observe({ type: 'first-input', buffered: true });
  }

  measureCLS() {
    let clsValue = 0;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
        }
      }
      this.metrics.cls = clsValue;
      this.logMetric('CLS', clsValue);
    }).observe({ type: 'layout-shift', buffered: true });
  }

  measureResourceTiming() {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.initiatorType === 'img') {
          this.trackImageLoad(entry);
        } else if (entry.initiatorType === 'fetch' || entry.initiatorType === 'xmlhttprequest') {
          this.trackAPICall(entry);
        }
      }
    }).observe({ type: 'resource', buffered: true });
  }

  trackImageLoad(entry) {
    const loadTime = entry.responseEnd - entry.startTime;
    if (!this.metrics.imageLoads) this.metrics.imageLoads = [];
    this.metrics.imageLoads.push({
      name: entry.name,
      loadTime,
      size: entry.transferSize
    });
    
    if (loadTime > 1000) {
      console.warn(`Slow image load: ${entry.name} took ${loadTime.toFixed(2)}ms`);
    }
  }

  trackAPICall(entry) {
    const duration = entry.responseEnd - entry.startTime;
    if (!this.metrics.apiCalls) this.metrics.apiCalls = [];
    this.metrics.apiCalls.push({
      url: entry.name,
      duration,
      size: entry.transferSize,
      cached: entry.transferSize === 0
    });
    
    if (duration > 2000) {
      console.warn(`Slow API call: ${entry.name} took ${duration.toFixed(2)}ms`);
    }
  }

  measureUserInteractions() {
    // Track click-to-response time
    let clickStartTime = null;
    
    document.addEventListener('click', () => {
      clickStartTime = performance.now();
    });

    // Track search response time
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const start = performance.now();
      try {
        const response = await originalFetch(...args);
        const duration = performance.now() - start;
        
        if (clickStartTime && args[0].includes('/api/search')) {
          const totalDuration = performance.now() - clickStartTime;
          this.trackSearchPerformance(totalDuration, duration);
          clickStartTime = null;
        }
        
        return response;
      } catch (error) {
        const duration = performance.now() - start;
        this.trackFailedRequest(args[0], duration);
        throw error;
      }
    };
  }

  trackSearchPerformance(totalDuration, networkDuration) {
    if (!this.metrics.searchPerformance) this.metrics.searchPerformance = [];
    this.metrics.searchPerformance.push({
      totalDuration,
      networkDuration,
      renderDuration: totalDuration - networkDuration,
      timestamp: Date.now()
    });
    
    console.log(`Search performance: ${totalDuration.toFixed(2)}ms total (${networkDuration.toFixed(2)}ms network)`);
  }

  trackFailedRequest(url, duration) {
    if (!this.metrics.failedRequests) this.metrics.failedRequests = [];
    this.metrics.failedRequests.push({
      url,
      duration,
      timestamp: Date.now()
    });
  }

  measureMemoryUsage() {
    if ('memory' in performance) {
      setInterval(() => {
        const memory = performance.memory;
        this.metrics.memoryUsage = {
          used: memory.usedJSHeapSize,
          total: memory.totalJSHeapSize,
          limit: memory.jsHeapSizeLimit,
          timestamp: Date.now()
        };
        
        // Warn if memory usage is high
        const usagePercent = (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100;
        if (usagePercent > 80) {
          console.warn(`High memory usage: ${usagePercent.toFixed(1)}%`);
        }
      }, 10000); // Check every 10 seconds
    }
  }

  logMetric(name, value) {
    console.log(`${name}: ${value.toFixed(2)}ms`);
    
    // Send to analytics service if configured
    if (window.gtag) {
      window.gtag('event', 'timing_complete', {
        name: name,
        value: Math.round(value)
      });
    }
  }

  getMetrics() {
    return this.metrics;
  }

  generateReport() {
    const report = {
      coreWebVitals: {
        fcp: this.metrics.fcp,
        lcp: this.metrics.lcp,
        fid: this.metrics.fid,
        cls: this.metrics.cls
      },
      performance: {
        averageImageLoadTime: this.getAverageImageLoadTime(),
        averageApiCallTime: this.getAverageApiCallTime(),
        averageSearchTime: this.getAverageSearchTime(),
        failedRequestsCount: this.metrics.failedRequests?.length || 0
      },
      memory: this.metrics.memoryUsage,
      timestamp: Date.now()
    };
    
    return report;
  }

  getAverageImageLoadTime() {
    if (!this.metrics.imageLoads || this.metrics.imageLoads.length === 0) return 0;
    const total = this.metrics.imageLoads.reduce((sum, load) => sum + load.loadTime, 0);
    return total / this.metrics.imageLoads.length;
  }

  getAverageApiCallTime() {
    if (!this.metrics.apiCalls || this.metrics.apiCalls.length === 0) return 0;
    const total = this.metrics.apiCalls.reduce((sum, call) => sum + call.duration, 0);
    return total / this.metrics.apiCalls.length;
  }

  getAverageSearchTime() {
    if (!this.metrics.searchPerformance || this.metrics.searchPerformance.length === 0) return 0;
    const total = this.metrics.searchPerformance.reduce((sum, search) => sum + search.totalDuration, 0);
    return total / this.metrics.searchPerformance.length;
  }

  // Performance optimization suggestions
  getOptimizationSuggestions() {
    const suggestions = [];
    
    if (this.metrics.lcp > 2500) {
      suggestions.push('Consider optimizing Largest Contentful Paint (LCP) - images or fonts may be loading slowly');
    }
    
    if (this.metrics.fid > 100) {
      suggestions.push('First Input Delay (FID) is high - consider code splitting or reducing JavaScript execution time');
    }
    
    if (this.metrics.cls > 0.1) {
      suggestions.push('Cumulative Layout Shift (CLS) is high - ensure images have dimensions and avoid dynamic content insertion');
    }
    
    const avgImageLoad = this.getAverageImageLoadTime();
    if (avgImageLoad > 1000) {
      suggestions.push('Images are loading slowly - consider image optimization, lazy loading, or CDN usage');
    }
    
    const avgApiCall = this.getAverageApiCallTime();
    if (avgApiCall > 2000) {
      suggestions.push('API calls are slow - consider caching, request optimization, or backend improvements');
    }
    
    if (this.metrics.memoryUsage && this.metrics.memoryUsage.used > this.metrics.memoryUsage.limit * 0.7) {
      suggestions.push('High memory usage detected - consider reducing component complexity or fixing memory leaks');
    }
    
    return suggestions;
  }
}

// Initialize performance monitoring
let performanceMonitor = null;

export function initPerformanceMonitoring() {
  if (!performanceMonitor && typeof window !== 'undefined') {
    performanceMonitor = new PerformanceMonitor();
    
    // Add to window for debugging
    window.performanceMonitor = performanceMonitor;
  }
  return performanceMonitor;
}

export function getPerformanceReport() {
  return performanceMonitor?.generateReport();
}

export function getOptimizationSuggestions() {
  return performanceMonitor?.getOptimizationSuggestions();
}

// React hook for performance metrics
export function usePerformanceMetrics() {
  const [metrics, setMetrics] = React.useState({});
  
  React.useEffect(() => {
    if (!performanceMonitor) {
      initPerformanceMonitoring();
    }
    
    const interval = setInterval(() => {
      if (performanceMonitor) {
        setMetrics(performanceMonitor.getMetrics());
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);
  
  return metrics;
}