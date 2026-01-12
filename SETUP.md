# 🚀 Setup Guide - Butterfly Classifier

Complete installation and configuration guide.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Common Issues](#common-issues)

---

## 💻 System Requirements

### Minimum Requirements

- **OS**: Windows 10+, macOS 10.15+, Ubuntu 20.04+
- **Python**: 3.12 or higher
- **RAM**: 2 GB available
- **Disk Space**: 2 GB free
- **Internet**: For initial package download

### Recommended Requirements

- **RAM**: 4 GB+
- **CPU**: Multi-core processor
- **Disk**: SSD for faster loading
- **Display**: 1920x1080 or higher

---

## 🔧 Installation Steps

### 1. Install Python

**Check if Python is installed:**
```bash
python --version
# or
python3 --version
```

**Should show**: Python 3.12.x

**If not installed:**
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python@3.12`
- **Linux**: `sudo apt install python3.12`

---

### 2. Download Project

```bash
# Option A: Git clone
git clone https://github.com/arju10/butterfly-classification.git
cd butterfly-classification

# Option B: Download ZIP
# Unzip to desired location
cd butterfly-classification
```

---

### 3. Create Virtual Environment (Recommended)

```bash
# Create venv
python -m venv venv

# Activate
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# You should see (venv) in terminal
```

---

### 4. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This will install:
# - TensorFlow 2.19.0 (~450 MB)
# - Streamlit 1.40.1 (~40 MB)
# - Plotly 5.24.1 (~10 MB)
# - Supporting libraries

# Total time: 2-5 minutes
```

---

### 5. Verify Installation

```bash
# Check all packages installed
pip list | grep -E "tensorflow|streamlit|plotly"

# Should show:
# tensorflow     2.19.0
# streamlit      1.52.2
# plotly         5.24.1
```

---

### 6. Verify Model File

```bash
# Check model exists
ls -lh models/butterfly_model_WORKING.keras

# Should show:
# -rw-r--r-- 1 user group 12.9M Jan 8 butterfly_model_WORKING.keras

# Check class indices
ls class_indices.json

# Should exist
```

---

### 7. Run Application

```bash
streamlit run streamlit_app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# 
#   Local URL: http://localhost:8501
#   Network URL: http://192.168.x.x:8501
```

Browser should open automatically!

---

## ⚙️ Configuration

### Custom Port

```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502

# Access: http://localhost:8502
```

### Disable Browser Auto-open

```bash
streamlit run streamlit_app.py --server.headless true

# Then manually open: http://localhost:8501
```

### Production Settings

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"
```

---

## ✅ Verification Checklist

### Installation Checklist

- [ ] Python 3.12+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list` shows all packages)
- [ ] Model file exists and is 12.9 MB
- [ ] class_indices.json exists
- [ ] Streamlit starts without errors

### Functionality Checklist

- [ ] App opens in browser
- [ ] "Model loaded successfully" message appears
- [ ] Can upload image
- [ ] "Identify Species" button appears
- [ ] Clicking button shows prediction
- [ ] Confidence gauge displays
- [ ] Top-5 chart shows

---

## 🐛 Common Issues

### Issue: "command not found: python"

**Solution:**
```bash
# Try python3
python3 --version

# Or install Python
# Windows: Download from python.org
# Mac: brew install python
# Linux: sudo apt install python3
```

---

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate  # or venv\Scripts\activate

# Reinstall
pip install tensorflow==2.19.0
```

---

### Issue: "Model not found"

**Solution:**
```bash
# Check file location
pwd  # Should be in butterfly_classifier directory

# Check model exists
ls models/butterfly_model_WORKING.keras

# If missing, you need the model file
```

---

### Issue: Port 8501 already in use

**Solution:**
```bash
# Option 1: Use different port
streamlit run streamlit_app.py --server.port 8502

# Option 2: Kill process on 8501
# Linux/Mac:
lsof -i :8501
kill -9 <PID>

# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

---

### Issue: "SSL Certificate Error"

**Solution:**
```bash
# Upgrade pip and certificates
pip install --upgrade pip certifi

# Reinstall packages
pip install --force-reinstall -r requirements.txt
```

---

### Issue: Slow Loading

**Causes:**
- First run (loading TensorFlow)
- Slow CPU
- Insufficient RAM

**Solutions:**
- Wait 10-15 seconds on first run
- Close other applications
- Restart computer if RAM is full

---

## 🔄 Updating

### Update Dependencies

```bash
# Activate venv
source venv/bin/activate

# Update pip
pip install --upgrade pip

# Update packages
pip install --upgrade tensorflow streamlit plotly

# Or reinstall from requirements
pip install -r requirements.txt --upgrade
```

### Update Application

```bash
# If using git
git pull origin main

# Or download new version manually
```

---

## 🗑️ Uninstallation

### Remove Virtual Environment

```bash
# Deactivate
deactivate

# Remove directory
rm -rf venv  # Linux/Mac
# or
rmdir /s venv  # Windows
```

### Remove Application

```bash
# Navigate to parent directory
cd ..

# Remove project
rm -rf butterfly_classifier
```

---

## 📊 System Resource Usage

### During Startup

- **CPU**: 50-80% for 5-10 seconds
- **RAM**: ~500 MB
- **Disk I/O**: Reading model file (12.9 MB)

### During Normal Use

- **CPU**: 5-10% idle, 30-50% during prediction
- **RAM**: ~800 MB
- **Network**: Minimal (only for Streamlit updates)

### Per Prediction

- **Time**: 0.5-1 second
- **CPU Spike**: 50-70%
- **RAM**: +50 MB temporary

---

## 🎯 Next Steps

After successful installation:

1. ✅ Test with sample butterfly images
2. ✅ Read main README.md for usage guide
3. ✅ Check DEPLOYMENT.md for production setup
4. ✅ Review TROUBLESHOOTING.md for common issues

---

## 💡 Pro Tips

### Faster Startup

```bash
# Keep model loaded (don't restart app)
# Use Streamlit's auto-reload for code changes
```

### Better Performance

```bash
# Use SSD for model storage
# Close unnecessary applications
# Allocate more RAM if available
```

### Development Mode

```bash
# Enable debug mode
streamlit run streamlit_app.py --logger.level debug

# See detailed logs
```

---

**Setup complete! Ready to classify butterflies! 🦋**
