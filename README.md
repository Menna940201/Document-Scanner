# 📄 AI Document Scanner & OCR Web App

An interactive web application built with **Python, OpenCV, and Streamlit** that automatically detects document corners from a skewed or perspective image, straightens (warps) the document, and extracts its text content using **Tesseract OCR**.

---

## Features
* **Perspective Transformation:** Automatically finds the 4 corners of a document and corrects the angle.
* **Interactive Adjustments:** Dynamic sliders to adjust Canny Edge Detection thresholds for low-contrast images.
* **OCR Text Extraction:** Uses Tesseract OCR to digitize and extract raw text from the scanned output.
* **Streamlit Web UI:** Clean, user-friendly, and modern English interface.

---

## Tech Stack & Libraries
* **Python 3.x**
* **Streamlit** (Web Framework)
* **OpenCV** (Computer Vision & Image Processing)
* **NumPy** (Matrix Manipulations)
* **PyTesseract** (Tesseract OCR wrapper)

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone <your-github-repo-link>
cd <your-folder-name>
```

### 2. Install Dependencies
Make sure you have the required libraries installed:
```bash
pip install streamlit opencv-python numpy pytesseract
```

### 3. Install Tesseract OCR Engine
* **Windows:** Download and install the installer from UB Mannheim. Update the path in `doc.py` if necessary:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```
* **Mac:** Install via Homebrew: `brew install tesseract`
* **Linux:** Install via apt: `sudo apt install tesseract-ocr`

### 4. Start the Application
Run the Streamlit app from your terminal:
```bash
streamlit run doc.py
```

---

## How it Works
1. **Pre-processing:** The image is converted to grayscale, blurred, and processed via Canny Edge Detection to highlight outlines.
2. **Contour Detection:** The largest 4-sided polygon (contour) with an area greater than 5000 (or customized threshold) is selected.
3. **Corner Reordering:** The 4 corners are sorted chronologically `[top-left, top-right, bottom-left, bottom-right]`.
4. **Warp Perspective:** `cv2.getPerspectiveTransform` and `cv2.warpPerspective` align the document flat.
5. **OCR:** Tesseract processes the final flat image and outputs the text block.
