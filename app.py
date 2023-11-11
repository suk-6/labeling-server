import os
from flask import Flask, jsonify, request
import base64

app = Flask(__name__)

imageCount = 0

if not os.path.exists("images"):
    os.mkdir("images")
else:
    confirm = input("Remove all files in images folder? (y/n): ")
    if confirm == "y":
        for file in os.listdir("images"):
            os.remove(f"images/{file}")
    else:
        imageCount = len(os.listdir("images"))


@app.route("/")
def index():
    return jsonify({"message": "OK"})


@app.route("/api/image", methods=["POST"])
def image():
    try:
        global imageCount
        data = request.get_json()
        slider = data["slider"]
        base64Image = data["image"]
        imageExtension = data["extension"]
        print(slider)

        with open(f"images/{imageCount}.{slider}.{imageExtension}", "wb") as f:
            f.write(base64.b64decode(base64Image))

        imageCount += 1

        return jsonify({"message": "OK"})

    except Exception as e:
        print(e)
        return jsonify({"message": "Error"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10001)
