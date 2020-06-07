from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd




app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/result', methods = ['POST']) 
def result(): 
	if request.method == 'POST': 
		to_predict_list = request.form.to_dict() 
		to_predict_list = list(to_predict_list.values()) 
		to_predict_list = list(map(int, to_predict_list)) 
		result = ValuePredictor(to_predict_list)		 
		if int(result)== 1: 
			prediction ='Purchased'
		else: 
			prediction ='Not Purchased'			
		return render_template("result.html", prediction = prediction) 

# prediction function 
def ValuePredictor(to_predict_list): 
	to_predict = np.array(to_predict_list).reshape(1, 3) 
	loaded_model = pickle.load(open("model.pkl", "rb")) 
	result = loaded_model.predict(to_predict) 
	return result[0] 

if __name__ == "__main__":
    app.run(debug=True)    
 