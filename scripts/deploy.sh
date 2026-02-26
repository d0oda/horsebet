#!/usr/bin/env bash
# ==========================================
# UmaEdge — Deployment Script
# ==========================================
#
# Usage:
#   ./scripts/deploy.sh --frontend    # Deploy frontend to Vercel
#   ./scripts/deploy.sh --backend     # Deploy backend to Render
#   ./scripts/deploy.sh --all         # Deploy both

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

deploy_frontend() {
    echo ""
    echo "=============================================="
    echo "  Deploying Frontend to Vercel..."
    echo "=============================================="
    echo ""

    if ! command -v vercel &> /dev/null; then
        echo "⚠️  Vercel CLI not found. Install with:"
        echo "   npm i -g vercel"
        exit 1
    fi

    cd "$PROJECT_DIR/frontend"
    vercel --prod
    echo "✅ Frontend deployed!"
}

deploy_backend() {
    echo ""
    echo "=============================================="
    echo "  Deploying Backend..."
    echo "=============================================="
    echo ""

    # Option 1: Docker push (if using container registry)
    if command -v docker &> /dev/null; then
        echo "Building Docker image..."
        cd "$PROJECT_DIR"
        docker build -t umaedge-api .
        echo "✅ Docker image built: umaedge-api"
        echo ""
        echo "  To push to a registry:"
        echo "    docker tag umaedge-api your-registry/umaedge-api"
        echo "    docker push your-registry/umaedge-api"
    fi

    # Option 2: Render (if using render.yaml)
    echo ""
    echo "  For Render deployment:"
    echo "    1. Push to GitHub"
    echo "    2. Go to https://dashboard.render.com"
    echo "    3. Create New → Blueprint → select this repo"
    echo ""

    # Option 3: Railway
    if command -v railway &> /dev/null; then
        echo "  Railway detected. Deploy with:"
        echo "    cd $PROJECT_DIR && railway up"
    fi

    echo "✅ Backend deployment prepared!"
}

# Parse args
case "${1:-}" in
    --frontend)
        deploy_frontend
        ;;
    --backend)
        deploy_backend
        ;;
    --all)
        deploy_backend
        deploy_frontend
        ;;
    *)
        echo "Usage: ./scripts/deploy.sh [--frontend|--backend|--all]"
        echo ""
        echo "  --frontend   Deploy Next.js frontend to Vercel"
        echo "  --backend    Build & prepare API backend for deployment"
        echo "  --all        Deploy both frontend and backend"
        exit 1
        ;;
esac
