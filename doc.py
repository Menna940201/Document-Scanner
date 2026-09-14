import cv2
import numpy as np
import pytesseract
import streamlit as st

###################################
widthImg=540
heightImg =640
#####################################

# cap = cv2.VideoCapture(1)
# cap.set(10,150)

# img1 = cv2.imread("Resources/1.jpg")
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

st.set_page_config(
    page_title="AI Document Scanner & OCR", page_icon="📄", layout="wide"
)

st.title("📄 AI Document Scanner with OCR Extraction")
st.write(
    "Upload a skewed or perspective image of a document to automatically scan, warp, and extract its text content."
)

st.sidebar.header("Configuration Settings")
widthImg = st.sidebar.slider("Target Image Width", 300, 1000, 540)
heightImg = st.sidebar.slider("Target Image Height", 400, 1200, 640)

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>1000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area >maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder (myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    #print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print("NewPoints",myPointsNew)
    return myPointsNew

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))

    return imgCropped


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

uploaded_file = st.file_uploader(
    "Choose a document image...", type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img1 = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img1,(widthImg,heightImg))
    imgContour = img.copy()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image Input")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    if biggest.size !=0:
        imgWarped=getWarp(img,biggest)
        with col2:
            st.subheader("Scanned Output (Warped)")
            st.image(
                cv2.cvtColor(imgWarped, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
        st.success("Document Detected Successfully!")
        # imageArray = ([img,imgThres],[imgContour,imgWarped])
        # imageArray = ([imgContour, imgWarped])
        # cv2.imshow("ImageWarped", imgWarped)
        st.header("🔍 Extracted Text Content")
        with st.spinner("Running OCR Text Recognition..."):
            try:
                boxes = pytesseract.image_to_data(imgWarped)
                extracted_words = []
                for x, b in enumerate(boxes.splitlines()):
                    if x != 0:
                        b = b.split()
                        if len(b) == 12:
                            if b[-1] != "-1" and b[-1].strip() != "":
                                extracted_words.append(b[-1])
                full_text = " ".join(extracted_words)

                if full_text.strip():
                    st.text_area(
                        "Recognized Text:", value=full_text, height=250
                    )
                else:
                    st.warning(
                        "No clear text detected. Try adjust standard orientation or brightness."
                    )
            except Exception as e:
                st.error(
                    f"OCR Error: Please ensure Tesseract OCR is correctly configured on your host environment. Details: {e}"
                )

    else:
        st.error(
            "Could not detect 4 document corners. Try another photo with clearer edges or high contrast background."
        )

# else:
#     imageArray = ([img, imgThres],[img, img])
#     imageArray = ([imgContour, img])
# stackedImages = stackImages(0.6,imageArray)
# cv2.imshow("WorkFlow", stackedImages)

cv2.waitKey(0)