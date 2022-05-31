import pickle
from flask import Flask,request,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app = Flask(__name__)
Random_forest_model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home_page():
    'html page for taking inputs via web'
    return render_template('home.html')

@app.route('/prdiction_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = Random_forest_model.predict(new_data)[0]
    return jsonify(output)


@app.route('/predict_web', methods=['POST'])
def predict_web():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output = Random_forest_model.predict(final_features)[0]
    print(output)
    # output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))


if __name__ == "__main__":
    app.run(debug=True,port=8000)