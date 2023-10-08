from flask import Flask,request,jsonify
import pickle,numpy as np
from flask_cors import CORS,cross_origin
app=Flask(__name__)
CORS(app)
global datum
global y
@app.route('/',methods=['GET'])
@cross_origin()
def wow():return jsonify({'msg':'hello we are using flask'})
@app.route('/prediction',methods=['POST'])
@cross_origin()
def result():
        M=request.get_json();I=M.get('arr',0);print(I);C=np.array(I);C=C.reshape(1,-1);print(C)
        with open('careerlast.pkl','rb')as N:J=pickle.load(N)
        F=J.predict(C);print(F);O=F[0];print(O);E=J.predict_proba(C);print(E);E=E>.05;P=0;D=0;A=0;G={};B={}
        while D<17:
                if E[P,D]:G[A]=D;A+=1
                D+=1
        print(D);print(G);A=0
        for(K,L)in G.items():
                if L!=F[0]:B[A]=L;print('final_res[index]:',B[A]);A+=1
                print(B)
        Q={0:'AI ML Specialist',1:'API Integration Specialist',2:'Application Support Engineer',3:'Business Analyst',4:'Customer Service Executive',5:'Cyber Security Specialist',6:'Data Scientist',7:'Database Administrator',8:'Graphics Designer',9:'Hardware Engineer',10:'Helpdesk Engineer',11:'Information Security Specialist',12:'Networking Engineer',13:'Project Manager',14:'Software Developer',15:'Software Tester',16:'Technical Writer'};A=1;print(B);H=[]
        for K in B:H.append(Q[B[K]])
        print(H);return jsonify({'role':H})
if __name__=='__main__':app.run(debug=True, host="0.0.0.0", port=80)


