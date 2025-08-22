#!/usr/bin/env python3
"""
Development Environment Setup Script for AIC25
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_prerequisites():
    """Check if required tools are installed."""
    print("🔍 Checking prerequisites...")
    
    required_tools = [
        ("python3", "Python 3.12+"),
        ("node", "Node.js 18+"),
        ("npm", "NPM"),
        ("ffmpeg", "FFmpeg"),
        ("git", "Git")
    ]
    
    missing_tools = []
    for tool, description in required_tools:
        if not shutil.which(tool):
            missing_tools.append(f"{tool} ({description})")
        else:
            print(f"✅ {description} found")
    
    if missing_tools:
        print("❌ Missing required tools:")
        for tool in missing_tools:
            print(f"  - {tool}")
        return False
    
    return True

def setup_python_environment():
    """Setup Python development environment."""
    print("\n📦 Setting up Python environment...")
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing AIC25 package"):
        return False
    
    # Install development dependencies
    if not run_command("pip install pytest black isort mypy", "Installing development dependencies"):
        return False
    
    return True

def setup_frontend():
    """Setup frontend development environment."""
    print("\n🌐 Setting up frontend environment...")
    
    web_dir = Path("web")
    if not web_dir.exists():
        print(f"❌ Web directory not found: {web_dir}")
        return False
    
    os.chdir(web_dir)
    
    # Install NPM dependencies
    if not run_command("npm install", "Installing NPM dependencies"):
        return False
    
    os.chdir("..")
    return True

def create_dev_workspace():
    """Create development workspace."""
    print("\n📁 Setting up development workspace...")
    
    workspace_dir = Path("dev_workspace")
    if workspace_dir.exists():
        print(f"⚠️ Development workspace already exists: {workspace_dir}")
        return True
    
    # Initialize workspace
    if not run_command(f"aic25-cli init {workspace_dir}", "Creating development workspace"):
        return False
    
    # Create sample directory structure
    sample_dirs = [
        "videos",
        "keyframes", 
        "features",
        "indices"
    ]
    
    for dir_name in sample_dirs:
        (workspace_dir / dir_name).mkdir(exist_ok=True)
    
    print(f"✅ Development workspace created at: {workspace_dir}")
    return True

def setup_environment_file():
    """Create .env file for development."""
    print("\n⚙️ Setting up environment configuration...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️ .env file already exists")
        return True
    
    env_content = """# AIC25 Development Environment
AIC25_WORK_DIR=./dev_workspace
AIC25_CONFIG_ENV=development
AIC25_HOST=0.0.0.0
AIC25_PORT=8000
AIC25_DEBUG=true

# Frontend development
VITE_API_BASE_URL=http://localhost:8000
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with development settings")
    return True

def print_next_steps():
    """Print next steps for the developer."""
    print("\n🎉 Development environment setup complete!")
    print("\n📋 Next steps:")
    print("1. Add some videos to your workspace:")
    print("   aic25-cli add path/to/your/videos")
    print("\n2. Process and index the videos:")
    print("   aic25-cli analyse")
    print("   aic25-cli index")
    print("\n3. Start the backend server:")
    print("   aic25-cli serve")
    print("\n4. In another terminal, start the frontend:")
    print("   cd web && npm run dev")
    print("\n5. Open http://localhost:3000 in your browser")
    print("\n📚 Documentation:")
    print("   - docs/development/setup.md")
    print("   - docs/user-guide/installation.md")

def main():
    """Main setup function."""
    print("🚀 AIC25 Development Environment Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the AIC25 root directory")
        sys.exit(1)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Please install missing prerequisites and try again")
        sys.exit(1)
    
    # Setup steps
    steps = [
        setup_python_environment,
        setup_frontend,
        create_dev_workspace,
        setup_environment_file
    ]
    
    for step in steps:
        if not step():
            print(f"\n❌ Setup failed at step: {step.__name__}")
            sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()