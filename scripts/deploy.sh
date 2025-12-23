#!/bin/bash
# Deploy MT service to Modal.com
#
# Usage:
#   ./scripts/deploy.sh [--download-model]
#
# Options:
#   --download-model  Download and cache model before deploying

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Unitra MT Service Deployment"
echo "=========================================="

# Check for Modal CLI
if ! command -v modal &> /dev/null; then
    echo "Error: Modal CLI not found. Install with: pip install modal"
    exit 1
fi

# Check Modal authentication
echo ""
echo "Checking Modal authentication..."
if ! modal token show &> /dev/null; then
    echo "Error: Not authenticated with Modal. Run: modal token new"
    exit 1
fi
echo "✓ Modal authenticated"

# Optional: Download model first
if [[ "$1" == "--download-model" ]]; then
    echo ""
    echo "Downloading model to persistent volume..."
    modal run scripts/download_model.py
    echo "✓ Model downloaded"
fi

# Deploy the service
echo ""
echo "Deploying MT service..."
modal deploy src/services/mt_service.py

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Web endpoints:"
echo "  POST /translate - Translation endpoint"
echo "  GET  /health    - Health check"
echo ""
echo "Test with:"
echo "  curl -X POST https://unitra--unitra-mt-translate.modal.run \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\": \"Hello, world!\", \"source_lang\": \"en\", \"target_lang\": \"zh\"}'"
echo ""
