# Development Setup Guide

This guide helps you set up a development environment for AIC25.

## Prerequisites

- Python 3.12+
- Node.js 18+
- Git
- FFmpeg (for video processing)

## Quick Setup

### 1. Clone and Setup Backend

```bash
git clone https://github.com/DacPhu/AIC25.git
cd AIC25

# Install Python dependencies
pip install -e .

# Initialize development workspace
aic25-cli init ./dev_workspace
cd dev_workspace
```

### 2. Setup Frontend

```bash
cd ../web
npm install
npm run dev
```

### 3. Start Development Servers

**Terminal 1 - Backend:**
```bash
cd /path/to/AIC25
export AIC25_WORK_DIR=./dev_workspace
export AIC25_CONFIG_ENV=development
aic25-cli serve --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd /path/to/AIC25/web
npm run dev
```

## Development Workflow

### Backend Development

1. **Code Location**: `src/aic25/`
2. **Configuration**: Use `config/development.yaml`
3. **Testing**: `pytest src/tests/`
4. **Linting**: `black src/ && isort src/`

### Frontend Development

1. **Code Location**: `web/src/`
2. **Development Server**: `npm run dev` (runs on port 3000)
3. **Type Checking**: `npm run type-check`
4. **Linting**: `npm run lint`
5. **Building**: `npm run build`

### Adding New Features

1. **Backend API**: Add routes in `src/aic25/api/`
2. **Frontend UI**: Add components in `web/src/components/`
3. **Services**: Add business logic in `src/aic25/services/`

## Project Structure

```
AIC25/
├── src/aic25/          # Python backend
├── web/                # React frontend  
├── config/             # Configuration files
├── docs/               # Documentation
├── scripts/            # Build/deployment scripts
└── tests/              # Test files
```

## Configuration

The system uses YAML configuration files in the `config/` directory:

- `default.yaml`: Base configuration
- `development.yaml`: Development overrides
- `production.yaml`: Production overrides

Set environment variable `AIC25_CONFIG_ENV` to choose configuration.

## Debugging

### Backend Debugging
- Set `logging.level: DEBUG` in development config
- Use Python debugger: `import pdb; pdb.set_trace()`

### Frontend Debugging
- Use browser dev tools
- React dev tools extension
- Vite dev server provides hot reload

## Common Issues

### Port Conflicts
- Backend runs on port 8000
- Frontend runs on port 3000
- Change ports in respective config files if needed

### CORS Issues
- Backend allows all origins in development
- Check browser console for CORS errors

### Database Issues
- Ensure you've run `aic25-cli analyse` and `aic25-cli index`
- Check database configuration in `config/development.yaml`