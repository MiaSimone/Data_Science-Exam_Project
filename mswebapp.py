
# Import libraries
import numpy as np
import pandas as pd

# from sklearn.externals 
import joblib
import pickle
from flask import Flask, request, jsonify, render_template

# create an instance (our app)
#app = Flask(__name__)
app = Flask(__name__, template_folder='html_documents')

bcmodel = joblib.load('GaussianNB_model.sav')

def intToWord(n):
    # Getting the data from a CSV file and splits the data for ervery ','
    df_carList_make_word = pd.read_csv("datasets/carList.csv", delimiter=",")
    df_carList_make_word = df_carList_make_word['Make'].unique()
    words = pd.DataFrame(df_carList_make_word, columns=['Make'])
    
    df_carList_make_int = df_carList['Make'].unique()
    integers = pd.DataFrame(df_carList_make_int, columns=['Make_int'])
    
    df_stack = pd.concat([words, integers], axis=1)
    df_stack.columns = ['Make', 'Make_int']
    
    val = df_stack.query('Make_int  == '+str(n))['Make']
    
    return val


@app.route('/', methods=['GET', 'POST'])

@app.route('/hi/<name>')
def hello(name = None):
    return render_template('start.html', name=name)
# name is parameter in the template: {{name}}

@app.route('/predict')
def predict():
    return render_template('prediction.html')

@app.route('/predicted', methods=['GET', 'POST'])
def predicted():
    if request.method == 'POST':
        x1 = request.form['x1']
        x2 = request.form['x2']
        x3 = request.form['x3']
        x4 = request.form['x4']
        x5 = request.form['x5']
        x6 = request.form['x6']
        X = [[x1, x2, x3, x4, x5, x6]]
        predicted = bcmodel.predict(X)
        predictedWord = intToWord(predicted)
          
        return render_template("predicted.html", content=X, prediction=[predicted, predictedWord])
    
@app.route('/bye')
def bye():
    return render_template('bye.html')

if __name__ == '__main__':
    app.run(debug=True)
