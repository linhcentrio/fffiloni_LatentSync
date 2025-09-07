#!/bin/bash

# Build và test script cho LatentSync v1.0 Docker image
# Tối ưu hóa cho RunPod serverless

set -e

echo "🚀 Building LatentSync v1.0 Docker Image for RunPod Serverless"
echo "================================================================"

# Configuration
IMAGE_NAME="latentsync-v1.0"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Pre-build checks
print_step "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker không được cài đặt"
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    print_error "Dockerfile không tồn tại"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt không tồn tại"
    exit 1
fi

print_success "Prerequisites OK"

# Step 2: Clean up old images
print_step "Cleaning up old images..."
docker rmi "${FULL_IMAGE_NAME}" 2>/dev/null || true
docker system prune -f

# Step 3: Build image với optimizations
print_step "Building Docker image..."
echo "Image: ${FULL_IMAGE_NAME}"

# Build với BuildKit để tối ưu hóa
export DOCKER_BUILDKIT=1

docker build \
    --progress=plain \
    --tag "${FULL_IMAGE_NAME}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

print_success "Build completed"

# Step 4: Check image size
print_step "Checking image size..."
IMAGE_SIZE=$(docker images "${FULL_IMAGE_NAME}" --format "table {{.Size}}" | tail -n 1)
echo "Image size: ${IMAGE_SIZE}"

# Warning nếu image quá lớn (>8GB thường chậm trên serverless)
if [[ $(docker images "${FULL_IMAGE_NAME}" --format "{{.Size}}" | tail -n 1 | grep -E "[0-9]+(\.[0-9]+)?GB" | cut -d'G' -f1) > 8 ]]; then
    print_warning "Image size > 8GB - có thể chậm startup trên serverless"
fi

# Step 5: Comprehensive functionality test
print_step "Running LatentSync v1.0 core test suite..."

# Run comprehensive test script
docker run --rm "${FULL_IMAGE_NAME}" python test_latentsync_core.py

if [ $? -eq 0 ]; then
    print_success "LatentSync v1.0 core test suite passed"
else
    print_error "LatentSync v1.0 core test suite failed"
    exit 1
fi

# Step 6: Performance info
print_step "Gathering performance info..."
echo "Docker image layers:"
docker history "${FULL_IMAGE_NAME}" --format "table {{.CreatedBy}}\t{{.Size}}" | head -10

# Step 7: RunPod deployment info
print_step "RunPod deployment information..."
echo "📋 LatentSync v1.0 Core Deployment checklist:"
echo "   ✅ Image name: ${FULL_IMAGE_NAME}"
echo "   ✅ Exposed port: 8000"
echo "   ✅ Health check: Enabled"
echo "   ✅ CUDA support: Yes"
echo "   ✅ LatentSync v1.0 models: Pre-downloaded"
echo "   ✅ FFmpeg video processing: Included"
echo "   ❌ Face enhancement: Disabled (core only)"
echo ""
echo "🔧 RunPod configuration (optimized for v1.0 core):"
echo "   - Container Disk: Recommend 10GB+ (reduced size)"
echo "   - GPU: RTX3090/RTX4090/A100 recommended"
echo "   - Memory: 12GB+ recommended (reduced requirement)"
echo "   - Environment Variables:"
echo "     * RUNPOD_AI_API_KEY=<your_key>"
echo ""
echo "🎯 Features:"
echo "   ✅ LatentSync v1.0 core lipsync"
echo "   ✅ FFmpeg-based video processing"
echo "   ✅ Pingpong loop support"
echo "   ✅ Video cropping support"
echo "   ❌ GFPGAN face enhancement (removed)"
echo "   ❌ Face detection (removed)"
echo ""

# Step 8: Push instructions (optional)
echo -e "${BLUE}[INFO]${NC} To push to registry:"
echo "docker tag ${FULL_IMAGE_NAME} your-registry/${IMAGE_NAME}:${TAG}"
echo "docker push your-registry/${IMAGE_NAME}:${TAG}"

print_success "Build and test completed successfully! 🎉"
echo "Ready for RunPod serverless deployment."
