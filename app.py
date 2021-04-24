import torch

from feelpath.models import MLP
from flask import Flask, request
from flask_restful import Resource, Api

from feelpath.usage import predict
from feelpath.markov import viterbi_wrap


app = Flask(__name__)
api = Api(app)



MAX_LEN = 50
EMBEDDING_LEN = 100
NUM_CLASSES = 7
SAVE_PATH = "save/model.pt"

model = MLP(MAX_LEN, EMBEDDING_LEN, NUM_CLASSES)
model.load_state_dict(torch.load(SAVE_PATH,
                                    map_location=torch.device('cpu')) )
model.eval()
viterbi = viterbi_wrap()

class FeelPath(Resource):
    def get(self, data):
        sentence = request.form['data']
        pred = predict(model, sentence)
        path, _, _ = viterbi(pred)
        return {"emotions": pred, "next" : path.tolist()}

api.add_resource(FeelPath, '/<string:data>')

if __name__ == '__main__':
    app.run(debug=True)