from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

# loading model
model = pickle.load(open('model.pkl','rb'))

# create flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    sex = float(request.form['Sex'])
    hig = float(request.form['hig'])
    whw = float(request.form['whw'])
    len = float(request.form['len'])
    dim = float(request.form['dim'])
    sw = float(request.form['sw'])
    vw = float(request.form['vw'])
    shw = float(request.form['shw'])
    

    features = np.array([[sex,hig,whw,len,dim,sw,vw,shw]])
    prediction = model.predict(features).reshape(1,-1)

    return render_template('index.html', age = prediction[0])

#-Sex,Length,Diameter,Height,Whole_Weight,Shucked_Weight,Viscera_weight,Shell_weight
# python main
if __name__ == "__main__":
    app.run(debug=True)

