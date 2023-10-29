from flask import Flask, render_template, request, redirect, url_for, flash
import os
import base64

from Train.area import area_cal
from Train.testing import prediction

app = Flask(__name__)
app.secret_key = 'some_secret_key'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = 'uploaded_image.jpg'
            file_path = os.path.join('static', 'uploads', filename)
            file.save(file_path)
            predicted_class, confidence = prediction(file_path)
            print(predicted_class, confidence)
            total_leaf_area, disease_area, percentage_diseased = area_cal(file_path)
            return render_template('b.html', predicted_class=predicted_class, confidence = "{:.2f}%".format(confidence * 100), total_leaf_area=total_leaf_area, disease_area=disease_area, percentage_diseased= "{:.2f}%".format(percentage_diseased))

    return render_template('a.html')


if __name__ == '__main__':
    app.run(debug=True)
