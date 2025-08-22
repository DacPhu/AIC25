# AIC25 Project Restructuring Plan

## Current Issues
1. **Complex nested structure**: Frontend buried in `src/entry/web/view/`
2. **Mixed concerns**: Backend and frontend intermixed
3. **Poor organization**: CLI, API, and web UI all in confusing paths
4. **Configuration scatter**: Config files in multiple locations
5. **Unclear developer experience**: Hard to understand and contribute to

## Proposed New Structure

```
AIC25/
├── README.md                      # Main project documentation
├── LICENSE
├── pyproject.toml                 # Python project config
├── requirements.txt               # Python dependencies (if needed)
├── .gitignore
├── .env.example                   # Environment variables template
│
├── docs/                          # Centralized documentation
│   ├── api/                       # API documentation
│   ├── deployment/               # Deployment guides
│   ├── development/              # Development setup
│   └── user-guide/               # User documentation
│
├── config/                        # Configuration files
│   ├── default.yaml              # Default configuration
│   ├── development.yaml          # Development overrides
│   └── production.yaml           # Production settings
│
├── scripts/                       # Build and deployment scripts
│   ├── build.py                  # Build automation
│   ├── deploy.py                 # Deployment automation
│   └── dev-setup.py              # Development environment setup
│
├── src/                          # Python backend source
│   ├── aic25/                    # Main package
│   │   ├── __init__.py
│   │   ├── cli/                  # CLI commands
│   │   ├── api/                  # Web API (FastAPI)
│   │   ├── core/                 # Core business logic
│   │   ├── services/             # Service layer
│   │   ├── config/               # Configuration management
│   │   └── utils/                # Utilities
│   └── tests/                    # Python tests
│
├── web/                          # Frontend application (React)
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── hooks/
│   │   ├── utils/
│   │   └── types/
│   └── public/
│
├── docker/                       # Docker configurations
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yaml
│
└── workspace/                    # Default workspace (gitignored)
    ├── videos/
    ├── keyframes/
    ├── features/
    └── indices/
```

## Key Improvements

### 1. Clear Separation of Concerns
- **Backend**: `src/aic25/` - All Python code
- **Frontend**: `web/` - All React/TypeScript code
- **Config**: `config/` - All configuration
- **Docs**: `docs/` - All documentation

### 2. Better Developer Experience
- Clear entry points
- Consistent naming conventions
- Proper build scripts
- Development environment setup

### 3. Improved User Experience
- Simple installation commands
- Clear documentation structure
- Better error messages
- Development vs production configs

### 4. Professional Structure
- Industry-standard project layout
- Proper separation of frontend/backend
- Docker support for easy deployment
- Comprehensive testing structure

## Implementation Steps

1. **Move frontend to `/web/`**
2. **Reorganize backend in `/src/aic25/`**
3. **Centralize configuration in `/config/`**
4. **Consolidate documentation in `/docs/`**
5. **Create development tooling**
6. **Update build processes**
7. **Add Docker support**
8. **Update documentation**

## Benefits

- **Developers**: Easier to understand, contribute, and maintain
- **Users**: Simpler installation and clearer documentation
- **Deployment**: Better separation for containerization
- **Maintenance**: Cleaner code organization and testing