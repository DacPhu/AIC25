import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  
  // Path aliases for better imports
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/components': path.resolve(__dirname, './src/components'),
      '@/services': path.resolve(__dirname, './src/services'),
      '@/hooks': path.resolve(__dirname, './src/hooks'),
      '@/routes': path.resolve(__dirname, './src/routes'),
      '@/assets': path.resolve(__dirname, './src/assets'),
      '@/resources': path.resolve(__dirname, './src/resources'),
      '@/styles': path.resolve(__dirname, './src/styles')
    }
  },
  
  // Performance optimizations
  build: {
    // Build output directory - can be overridden by VITE_OUTDIR env var
    outDir: process.env.VITE_OUTDIR || 'dist',
    // Enable code splitting
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunks for better caching
          'react-vendor': ['react', 'react-dom'],
          'router-vendor': ['react-router-dom'],
          'utils-vendor': ['axios', 'classnames', 'jszip', 'localforage']
        }
      }
    },
    // Enable source maps for debugging
    sourcemap: true,
    // Optimize chunk size
    chunkSizeWarningLimit: 1000,
    // Enable minification
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true, // Remove console logs in production
        drop_debugger: true
      }
    }
  },
  
  // Development server optimizations
  server: {
    // Enable HMR optimizations
    hmr: {
      overlay: true
    }
  },
  
  // Dependency pre-bundling optimizations
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'axios',
      'classnames'
    ],
    // Force pre-bundling of these dependencies
    force: true
  }
})
