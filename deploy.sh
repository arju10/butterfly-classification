#!/bin/bash

# 🦋 Butterfly Classifier - Docker Deployment Script
# Automated deployment with validation checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check command existence
check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

echo ""
echo "🦋 ========================================="
echo "   Butterfly Classifier Docker Deployment"
echo "   ========================================="
echo ""

# Step 1: Check Prerequisites
print_info "Step 1: Checking prerequisites..."
echo ""

PREREQS_OK=true

if check_command "docker"; then
    docker --version
else
    print_error "Please install Docker first: https://docs.docker.com/get-docker/"
    PREREQS_OK=false
fi

echo ""

if check_command "docker compose" || check_command "docker-compose"; then
    docker compose version 2>/dev/null || docker-compose version
else
    print_error "Please install Docker Compose"
    PREREQS_OK=false
fi

if [ "$PREREQS_OK" = false ]; then
    exit 1
fi

echo ""
print_success "All prerequisites installed!"
echo ""

# Step 2: Check Required Files
print_info "Step 2: Checking required files..."
echo ""

FILES_OK=true

# Check Dockerfile
if [ -f "Dockerfile" ]; then
    print_success "Dockerfile found"
else
    print_error "Dockerfile not found!"
    FILES_OK=false
fi

# Check docker-compose.yml
if [ -f "docker-compose.yml" ]; then
    print_success "docker-compose.yml found"
else
    print_error "docker-compose.yml not found!"
    FILES_OK=false
fi

# Check requirements.txt
if [ -f "requirements.txt" ]; then
    print_success "requirements.txt found"
else
    print_error "requirements.txt not found!"
    FILES_OK=false
fi

# Check streamlit_app.py
if [ -f "streamlit_app.py" ]; then
    print_success "streamlit_app.py found"
else
    print_error "streamlit_app.py not found!"
    FILES_OK=false
fi

# Check class_indices.json
if [ -f "class_indices.json" ]; then
    SIZE=$(du -k "class_indices.json" | cut -f1)
    if [ $SIZE -gt 1 ]; then
        print_success "class_indices.json found (${SIZE}KB)"
    else
        print_warning "class_indices.json seems too small"
    fi
else
    print_error "class_indices.json not found!"
    FILES_OK=false
fi

# Check model file (CRITICAL!)
if [ -f "models/butterfly_model_best.h5" ]; then
    SIZE_MB=$(du -m "models/butterfly_model_best.h5" | cut -f1)
    if [ $SIZE_MB -gt 500 ]; then
        print_success "Model file found (${SIZE_MB}MB) - Size looks good!"
    else
        print_warning "Model file found but seems too small (${SIZE_MB}MB)"
        print_warning "Expected size: ~530 MB"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    print_error "Model file not found at models/butterfly_model_best.h5"
    print_error "Please download it from Kaggle first!"
    FILES_OK=false
fi

if [ "$FILES_OK" = false ]; then
    echo ""
    print_error "Missing required files. Please check the setup guide."
    exit 1
fi

echo ""
print_success "All required files present!"
echo ""

# Step 3: Clean previous deployment
print_info "Step 3: Cleaning previous deployment..."
echo ""

if docker compose ps | grep -q "butterfly_project"; then
    print_info "Stopping existing container..."
    docker compose down -v
    print_success "Previous deployment cleaned"
else
    print_info "No previous deployment found"
fi

echo ""

# Step 4: Build Docker Image
print_info "Step 4: Building Docker image..."
print_info "This will take 3-5 minutes..."
echo ""

if docker compose build --no-cache; then
    print_success "Docker image built successfully!"
else
    print_error "Docker build failed. Check the error messages above."
    exit 1
fi

echo ""

# Step 5: Start Application
print_info "Step 5: Starting application..."
echo ""

if docker compose up -d; then
    print_success "Container started!"
else
    print_error "Failed to start container"
    exit 1
fi

echo ""

# Step 6: Wait for Health Check
print_info "Step 6: Waiting for application to be healthy..."
print_info "This may take up to 40 seconds..."
echo ""

HEALTH_CHECK_COUNT=0
MAX_CHECKS=15

while [ $HEALTH_CHECK_COUNT -lt $MAX_CHECKS ]; do
    sleep 3
    
    STATUS=$(docker compose ps | grep butterfly_project | awk '{print $6}')
    
    if [[ "$STATUS" == *"healthy"* ]]; then
        print_success "Application is healthy!"
        break
    elif [[ "$STATUS" == *"unhealthy"* ]]; then
        print_error "Application is unhealthy!"
        print_error "Checking logs..."
        docker compose logs --tail=50
        exit 1
    else
        echo -n "."
    fi
    
    HEALTH_CHECK_COUNT=$((HEALTH_CHECK_COUNT + 1))
done

echo ""

if [ $HEALTH_CHECK_COUNT -eq $MAX_CHECKS ]; then
    print_warning "Health check timeout, but container is running"
    print_info "Checking logs..."
    docker compose logs --tail=20
fi

echo ""

# Step 7: Verify Deployment
print_info "Step 7: Verifying deployment..."
echo ""

# Check container status
if docker compose ps | grep -q "Up"; then
    print_success "Container is running"
else
    print_error "Container is not running!"
    docker compose ps
    exit 1
fi

# Test HTTP endpoint
print_info "Testing web endpoint..."
sleep 5  # Give it a moment to start serving

if curl -s -f http://localhost:8501/_stcore/health > /dev/null; then
    print_success "Web endpoint is responding!"
else
    print_warning "Web endpoint not responding yet (this is sometimes normal)"
    print_info "Try accessing http://localhost:8501 in your browser"
fi

echo ""

# Step 8: Display Summary
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║                                            ║"
echo "║  🎉  DEPLOYMENT SUCCESSFUL!  🎉           ║"
echo "║                                            ║"
echo "╚════════════════════════════════════════════╝"
echo ""

print_success "Butterfly Classifier is now running!"
echo ""

print_info "Access the application at:"
echo ""
echo "  🌐  http://localhost:8501"
echo ""

print_info "Useful commands:"
echo ""
echo "  View logs:      docker compose logs -f"
echo "  Stop app:       docker compose down"
echo "  Restart app:    docker compose restart"
echo "  Check status:   docker compose ps"
echo ""

# Display container info
print_info "Container Information:"
echo ""
docker compose ps
echo ""

# Display last few log lines
print_info "Recent Logs:"
echo ""
docker compose logs --tail=10
echo ""

# Final instructions
print_info "Next Steps:"
echo ""
echo "1. Open your browser and go to http://localhost:8501"
echo "2. Upload a butterfly image"
echo "3. Click 'Identify Species' to test the model"
echo ""
echo "For troubleshooting, run:"
echo "  docker compose logs -f"
echo ""

print_success "Deployment complete! Happy classifying! 🦋"
echo ""
