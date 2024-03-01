from flask import Flask, Response, request, jsonify
from main import extract_features, find_similar, model
app = Flask(__name__)

@app.route("/image_recommendation", methods = ["GET", "POST"])
def recommend():
    if request.method=="POST":
        data = request.json
        image = data["image"]
        image_feature= extract_features(image, model)
        similar, label = find_similar(image_feature)
        res = {
            "images":[i for i,j in similar],
            "label":[int(label)]
        }
        return jsonify(res)
    else:
        res = {"msg":"Please upload image"}
        return jsonify(res)
    
if __name__=="__main__":
    app.run(debug=True)