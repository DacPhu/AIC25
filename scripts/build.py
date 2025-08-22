#!/usr/bin/env python3
"""
Build Script for AIC25
Builds both backend and frontend for distribution.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        print(f"✅ {description} completed")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False, None

def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    artifacts_to_clean = [
        "build",
        "dist", 
        "*.egg-info",
        "web/dist",
        "web/node_modules/.cache"
    ]
    
    for pattern in artifacts_to_clean:
        if "*" in pattern:
            # Use shell expansion for patterns
            run_command(f"rm -rf {pattern}", f"Cleaning {pattern}")
        else:
            path = Path(pattern)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"✅ Cleaned {pattern}")

def build_frontend(mode="production"):
    """Build the frontend application."""
    print(f"\n🌐 Building frontend ({mode})...")
    
    web_dir = Path("web")
    if not web_dir.exists():
        print("❌ Web directory not found")
        return False
    
    # Install dependencies if needed
    if not (web_dir / "node_modules").exists():
        success, _ = run_command("npm install", "Installing frontend dependencies", cwd=web_dir)
        if not success:
            return False
    
    # Type check
    success, _ = run_command("npm run type-check", "Type checking frontend", cwd=web_dir)
    if not success:
        return False
    
    # Lint code
    success, _ = run_command("npm run lint", "Linting frontend code", cwd=web_dir)
    if not success:
        return False
    
    # Build for production
    build_cmd = "npm run build:prod" if mode == "production" else "npm run build"
    success, _ = run_command(build_cmd, f"Building frontend for {mode}", cwd=web_dir)
    if not success:
        return False
    
    return True

def build_backend():
    """Build the backend package."""
    print("\n📦 Building backend package...")
    
    # Check if setup tools are available
    success, _ = run_command("pip install --upgrade build setuptools wheel", "Updating build tools")
    if not success:
        return False
    
    # Run linting
    success, _ = run_command("black --check src/", "Checking code formatting")
    if not success:
        print("⚠️ Code formatting issues found. Run 'black src/' to fix.")
        return False
    
    success, _ = run_command("isort --check-only src/", "Checking import sorting")
    if not success:
        print("⚠️ Import sorting issues found. Run 'isort src/' to fix.")
        return False
    
    # Build package
    success, _ = run_command("python -m build", "Building Python package")
    if not success:
        return False
    
    return True

def copy_frontend_to_package():
    """Copy built frontend to the backend package."""
    print("\n📁 Integrating frontend with backend...")
    
    web_dist = Path("web/dist")
    if not web_dist.exists():
        print("❌ Frontend build not found. Build frontend first.")
        return False
    
    # Create target directory in the package
    package_web_dir = Path("src/aic25/web/static")
    package_web_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy frontend build to package
    shutil.copytree(web_dist, package_web_dir, dirs_exist_ok=True)
    print("✅ Frontend integrated with backend package")
    
    return True

def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    
    test_commands = [
        ("python -m pytest src/tests/ -v", "Running Python tests"),
        ("cd web && npm run test", "Running frontend tests")
    ]
    
    for cmd, description in test_commands:
        success, _ = run_command(cmd, description)
        if not success:
            print(f"⚠️ {description} failed, but continuing...")
    
    return True

def create_distribution():
    """Create final distribution files."""
    print("\n📦 Creating distribution...")
    
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    # Copy built artifacts
    artifacts = [
        ("dist/*.whl", "Python wheel"),
        ("dist/*.tar.gz", "Source distribution"),
        ("web/dist", "Frontend build")
    ]
    
    for src_pattern, description in artifacts:
        if "*" in src_pattern:
            success, _ = run_command(f"cp {src_pattern} dist/ 2>/dev/null || true", f"Copying {description}")
        else:
            src_path = Path(src_pattern)
            if src_path.exists():
                if src_path.is_dir():
                    shutil.copytree(src_path, dist_dir / src_path.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dist_dir)
                print(f"✅ Copied {description}")
    
    return True

def main():
    """Main build function."""
    parser = argparse.ArgumentParser(description="Build AIC25 for distribution")
    parser.add_argument("--mode", choices=["development", "production"], default="production",
                       help="Build mode")
    parser.add_argument("--frontend-only", action="store_true", 
                       help="Build only frontend")
    parser.add_argument("--backend-only", action="store_true",
                       help="Build only backend")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running tests")
    parser.add_argument("--skip-clean", action="store_true",
                       help="Skip cleaning build artifacts")
    
    args = parser.parse_args()
    
    print("🏗️ AIC25 Build Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the AIC25 root directory")
        sys.exit(1)
    
    # Clean build artifacts
    if not args.skip_clean:
        clean_build_artifacts()
    
    # Build steps
    success = True
    
    if not args.backend_only:
        if not build_frontend(args.mode):
            success = False
    
    if not args.frontend_only:
        if not build_backend():
            success = False
        
        # Integrate frontend with backend
        if not args.backend_only and not copy_frontend_to_package():
            success = False
    
    # Run tests
    if not args.skip_tests:
        run_tests()  # Don't fail build on test failures
    
    # Create distribution
    if success:
        create_distribution()
        print("\n🎉 Build completed successfully!")
        print(f"📁 Distribution files available in: {Path('dist').absolute()}")
    else:
        print("\n❌ Build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()