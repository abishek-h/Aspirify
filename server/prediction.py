from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS, cross_origin
from sklearn.metrics import accuracy_score

#idk whats happening

app = Flask(__name__)
CORS(app)

global datum
global y

@app.route('/',methods=['GET'])
@cross_origin()
def wow():
    return jsonify({
        "msg" : "hello we are using flask"
    })

@app.route('/prediction',methods = ['POST'])
@cross_origin()
def result():
    datum = request.get_json()
    arr = datum.get("arr",0)
    print(arr)

    data = np.array(arr)
    data = data.reshape(1,-1)
    print(data)

    with open("careerlast.pkl", 'rb') as f:
        loaded_model = pickle.load(f)
    predictions = loaded_model.predict(data)  
    
    print(predictions)
    reso = predictions[0]
    print(reso)
    pred = loaded_model.predict_proba(data)
    print(pred)
      #acc=accuracy_score(pred,)
    pred = pred > 0.05
      #print(predictions)
    i = 0
    j = 0
    index = 0
    res = {}
    final_res = {}
    while j < 17:
        if pred[i, j]:
            res[index] = j
            index += 1
        j += 1
    print(j)
    print(res)
    index = 0
    for key, values in res.items():
        if values != predictions[0]:
            final_res[index] = values
            print('final_res[index]:',final_res[index])
            index += 1
        print(final_res)
    jobs_dict = {0:'AI ML Specialist',
                   1:'API Integration Specialist',
                   2:'Application Support Engineer',
                   3:'Business Analyst',
                   4:'Customer Service Executive',
                   5:'Cyber Security Specialist',
                   6:'Data Scientist',
                   7:'Database Administrator',
                   8:'Graphics Designer',
                   9:'Hardware Engineer',
                   10:'Helpdesk Engineer',
                   11:'Information Security Specialist',
                   12:'Networking Engineer',
                   13:'Project Manager',
                   14:'Software Developer',
                   15:'Software Tester',
                   16:'Technical Writer'}
                
      #job[0] = jobs_dict[predictions[0]]
    index = 1

    print(final_res)
    finalarray = []
    for key in final_res:
        finalarray.append(jobs_dict[final_res[key]])
    print(finalarray)
    return jsonify({
        "role": finalarray
    })

if __name__ == '__main__':
   app.run(debug = True)