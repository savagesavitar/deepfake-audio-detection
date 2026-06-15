# 🚀 Quick Start Guide

## Step 1: Setup Google Colab (Recommended)

**No manual dataset download needed!** The notebook handles everything:

1. **Upload notebook to Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Upload `notebooks/deepfake_detection_colab.ipynb`

2. **Enable GPU:**
   - Runtime → Change runtime type → T4 GPU

3. **Get Kaggle API key:**
   - Go to [kaggle.com](https://kaggle.com) → Account → API → Create New API Token
   - Download the `kaggle.json` file

4. **Run the notebook:**
   - Run the first cell to upload `kaggle.json`
   - Dataset downloads automatically (~500MB)
   - Training starts automatically

**The notebook will:**
- ✅ Download dataset from Kaggle directly
- ✅ Extract and prepare data
- ✅ Train the CNN+LSTM model
- ✅ Evaluate and save models
- ✅ Allow you to download trained models

## Step 2: Install Dependencies

```bash
cd "B:\Deepfake Audio Detection"
pip install -r requirements.txt
```

## Step 3: Run the Notebook (Google Colab)

1. Upload `notebooks/deepfake_detection.ipynb` to Google Colab
2. Runtime → Change runtime type → GPU
3. Run all cells
4. Models will be saved to `models/` directory

## Step 4: Test the Model

```bash
# Create test samples
python create_test_samples.py

# Test with audio file
python test_audio.py --audio_path test_samples/sample_real.wav
```

## Step 5: Run Streamlit App

```bash
streamlit run app.py
```

Open browser to: http://localhost:8501

## Step 6: Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect repository
4. Deploy `app.py`

---

## 📊 Expected Results

| Metric | Target | Expected |
|--------|--------|----------|
| Accuracy | ≥ 80% | ~92% |
| EER | ≤ 12% | ~7.5% |
| F1 Score | ≥ 80% | ~91% |

---

## 📁 Project Files

- `notebooks/deepfake_detection.ipynb` - Main notebook
- `app.py` - Streamlit web app
- `test_audio.py` - Test script
- `README.md` - Full documentation
- `PERFORMANCE_REPORT.md` - Metrics report

---

## ❓ Need Help?

1. Check `README.md` for detailed documentation
2. Review `PERFORMANCE_REPORT.md` for metrics
3. Open issue on GitHub
