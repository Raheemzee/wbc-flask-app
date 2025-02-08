import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def count_wbc(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError("Error: Image not loaded. Check file path and format.")
    z = img.reshape((-1,3))
    z = np.float32(z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 3
    ret, label,center=cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    im_gray = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(im_gray, 5)
    ret, th1 = cv.threshold(img, 120, 50, cv.THRESH_BINARY)
    #gray = cv.cvtColor(th1, cv.COLOR_BGR2GRAY)
    binary = cv.Canny(th1, 50, 100)
    cnts, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_img = cv.drawContours(th1, cnts, -1, (0, 255, 0), 2) 
    wbc_count = len(cnts)

    uploaded_path = os.path.join(UPLOAD_FOLDER, "uploaded_image.jpg")
    processed_path = os.path.join(PROCESSED_FOLDER, "processed_image.jpg")
    cv.imwrite(processed_path, contour_img)
    
    return wbc_count, processed_path



@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
       
        if not file or file.filename == "":
            print("No file selected or uploaded.")
            return render_template("index.html", count=None, image_url=None, error="No file selected.")
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"File uploaded: {file_path}")
        file.save(file_path)

        if not os.path.exists(file_path):
            print(f"Error: Image was not saved - {file_path}")
            return render_template("index.html", count=None, image_url=None, error="File upload failed.")
        wbc_count, processed_image = count_wbc(file_path)
        
        if processed_image is None:
            print("Error processing image.")
            return render_template("index.html", count=None, image_url=None, error="Error processing image. Try a different one.")
        
        return render_template("index.html", count=wbc_count, image_url=url_for('processed_file', filename="processed_image.jpg"))
    
    return render_template("index.html", count=None, image_url=None)
    

@app.route("/processed/<filename>")
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)