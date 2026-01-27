# 🏥 Diabetic Retinopathy Detection System - Full Stack

## Quick Start (30 seconds!)

### ✨ Fastest Way to Run

### 📥 First Time: Download Models

If this is your first time, download the pre-trained models:

```powershell
python download_models.py
```

This downloads:
- `fusion_dr_model.keras` (277.8 MB) - Primary DR detection model
- `fusion_dr_model.h5` (276.8 MB) - Backup DR weights
- `generalist_model.h5` (Trained generalist model for multi-class classification)
- Models stored via Git LFS for efficient storage

**See `MODELS.md` for detailed information.**

**Double-click one of these:**
- 🟦 **Windows Batch:** `START_SYSTEM.bat`
- 🟦 **PowerShell:** `START_SYSTEM.ps1` (Right-click → Run with PowerShell)

**Or from terminal:**
```powershell
# PowerShell
.\START_SYSTEM.ps1

# OR Batch Command Prompt
START_SYSTEM.bat
```

---

## 📊 What You Get

| Component | URL | Status |
|-----------|-----|--------|
| **Web Dashboard** | http://localhost:3000 | ✅ Real-time UI |
| **AI Model API** | http://localhost:8001 | ✅ REST Endpoints |
| **DR Model File** | `fusion_dr_model.keras` | ✅ 53M Parameters |
| **Generalist Model** | `generalist_model.h5` | ✅ Multi-class classification |

---

## 🎯 How to Use

1. **Open Dashboard:** http://localhost:3000
2. **Upload Image:** Click upload or drag retinal fundus image
3. **Analyze:** Click "Analyze Image" button
4. **View Results:** See diagnosis with confidence scores

---

## 📁 Project Structure

```
.
├── src/                          # Backend (Python)
│   ├── api.py                   # Main API server
│   ├── model.py                 # Deep learning model
│   ├── preprocessing.py         # Image processing
│   └── ...
├── frontend/                     # Frontend (React)
│   ├── src/App.js              # Main UI component
│   ├── package.json            # Dependencies
│   └── ...
├── START_SYSTEM.bat            # Windows launcher
├── START_SYSTEM.ps1            # PowerShell launcher
├── SYSTEM_GUIDE.md             # Full documentation
└── debug_test.py               # Testing script
```

---

## 🔧 Manual Start (If Launchers Don't Work)

**Terminal 1 - Backend:**
```powershell
cd 'E:\Major project\DR_Detection_System'
.\venv\Scripts\python.exe -m uvicorn src.api:app --host localhost --port 8001
```

**Terminal 2 - Frontend:**
```powershell
cd 'E:\Major project\DR_Detection_System\frontend'
npm start
```

---

## 🐛 Troubleshooting

### Port 8001 Already In Use
```powershell
Get-NetTCPConnection -LocalPort 8001 | Stop-Process
```

### Cannot Find Python
Make sure you're using the virtual environment:
```powershell
.\venv\Scripts\python.exe --version
```

### npm not installed
Download from: https://nodejs.org/

### Frontend won't start
```powershell
cd frontend
npm install
npm start
```

---

## 📖 Documentation

- **Full Guide:** See `SYSTEM_GUIDE.md`
- **API Docs:** http://localhost:8001/docs (when running)
- **Code Files:** `src/api.py` has detailed docstrings

---

## ⚡ Performance

- **First Prediction:** ~30 seconds (model loads)
- **Next Predictions:** ~5-8 seconds
- **DR Accuracy:** 77.35% (validation set)
- **DR Classes:** 5 (No DR, Mild, Moderate, Severe, Proliferative)
- **Generalist Classes:** 4 (Normal, DR, Cataract, Glaucoma)

---

## 📊 Model Architecture

```
Input (224×224 RGB)
       ↓
    ┌──┴──┐
    ↓     ↓
  VGG  ResNet  DenseNet (Parallel Backbones)
    ↓     ↓     ↓
  Attention Blocks (Each Branch)
    ↓     ↓     ↓
  Project to 512 channels
    ↓─────┼─────↓
   Fusion Layer (Learned Weights)
       ↓
   Classification Head
       ↓
   5 Classes Output (DR Stages)
```

**Generalist Model:** Trained for multi-class classification (Normal, DR, Cataract, Glaucoma) using a similar architecture.

---

## 🚀 Next Steps

- [ ] Upload test images
- [ ] Verify predictions
- [ ] Check browser console (F12) for debugging
- [ ] Review backend logs for errors

---

## 💡 Tips

- **First use:** Takes 30-60 seconds to initialize
- **File format:** PNG, JPG, BMP supported
- **File size:** Max 10MB
- **Browser:** Chrome/Firefox recommended

---

## 📞 Quick Help

**API Endpoints:**
- `GET http://localhost:8001/health` - Server status
- `GET http://localhost:8001/model-info` - Model details
- `POST http://localhost:8001/predict` - Make prediction (file upload)

**Test Everything:**
```powershell
python debug_test.py
```

---

## 🎉 Ready to Go!

Your Diabetic Retinopathy Detection System is fully installed and ready to use.

**Next:** Open http://localhost:3000 in your browser! 🌐

---

**Version:** 1.1.0  
**Last Updated:** January 27, 2026  
**Status:** ✅ Production Ready
