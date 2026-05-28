import pytesseract
import os

# Set the path as in app.py
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    version = pytesseract.get_tesseract_version()
    print(f"SUCCESS: Tesseract found. Version: {version}")
    
    # Test path exists
    if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        print(f"Path verified: {pytesseract.pytesseract.tesseract_cmd}")
    else:
        print("ERROR: Final path does not exist on disk!")
        
except Exception as e:
    print(f"FAIL: Could not initialize Tesseract. Error: {str(e)}")
