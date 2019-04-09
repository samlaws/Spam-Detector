import pickle
import numpy as np
from sklearn.externals import joblib
from model import score, len_df, num_spam, num_ham
from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html', name='homepage')

@app.route('/about')
def about():
	my_score = score
	return render_template('about.html', name='aboutpage', score=my_score,
		total_am=len_df, spam=num_spam, ham=num_ham)

@app.route('/predict',methods=['POST'])
def predict():
	NB_spam_model = open('NB_spam_model.pkl','rb')
	spam_count_vectorizer = open('spam_count_vectorizer.pkl','rb')

	clf = joblib.load(NB_spam_model)
	cv = joblib.load(spam_count_vectorizer)


	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
		my_probability = clf.predict_proba(vect)
		my_spam_probability = np.round(my_probability[0][1] * 100, decimals=3)
		my_ham_probability = np.round(my_probability[0][0] * 100, decimals=3)


	return render_template('result.html',
	 						prediction = my_prediction,
							spam_probability = my_spam_probability,
							ham_probability = my_ham_probability)



if __name__ == '__main__':
	app.run(debug=True)
