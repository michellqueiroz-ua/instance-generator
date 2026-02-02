@echo off
echo Starting REQreate Web Interface...
call C:\Users\micqu\AppData\Local\miniconda3\Scripts\activate.bat
python -m streamlit run app.py
pause
