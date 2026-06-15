@echo off
echo ============================================
echo   Deepfake Audio Detection - Streamlit App
echo ============================================
echo.
echo Starting Streamlit app...
echo.
echo Open browser to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the app.
echo.
python -m streamlit run app.py --server.headless true
pause
