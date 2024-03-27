import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge regressor model and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html') ## It checks where the templates folder are

## lets check how to predict input data with ridge.pickle.
## inputs which we have to give to the model, we will specificaly write here.
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST' : #3 method is post, it will interact with pickle file and give the result with respect to that.
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled) ## this result is in list format, and there is only one element in the list

        return render_template('home.html',result=result[0]) ## it gives result values in home page also

    else:
        return render_template('home.html') 

if __name__=="__main__":
    app.run(host="0.0.0.0") ## it is maps with local address i.e. 127.0.0.1
