# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
from scan import scan_document

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    file = request.files['file']
    if file:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        scan_document(path)
        return redirect(url_for('results'))
    return 'No file uploaded', 400

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
