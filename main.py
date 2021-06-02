from flask import Flask, jsonify
from flask_restx import Api,Resource
from flask import request

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import pickle


app = Flask(__name__)
api = Api(app=app)
ns_raw = api.namespace('IsLeasable',description='if the product is leasable or not')

global model 
model = pickle.load(open('isleasable_model.sav', 'rb'))

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

@ns_raw.route("/")
class is_leasable(Resource):
    def post(self):
        data = request.json
        items = data['item_display_name']
        cleaned_items = [' '.join(clean(item)) for item in items]
        pred = model.predict(cleaned_items).tolist()
        prob = model.predict_proba(cleaned_items)
        prob = prob[:,1].tolist()
        prob = [1-x if x < 0.5 else x for x in prob]
        #return jsonify({"prediction":pred})
        return jsonify({"prediction":pred,"probability":prob})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
