############################################################
# mnister for Tensorflow (saved_model.pb) by PoodleMaster
############################################################
import argparse
import base64
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from flask_ngrok import run_with_ngrok
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


print("tensorflow : ", tf.__version__)
export_holder = 'export model/'

############################################################
# Flask
############################################################
app = Flask(__name__)


############################################################
# validation
############################################################
def validation():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--colab',
                        action="store_true",
                        help='Specify when running on Google Colab.(use flask-ngrok)')
    args = parser.parse_args()
    return args


############################################################
# 引数チェック
############################################################
args = validation()
if args.colab:                              # use Google Colab
    print("use flask-ngrok.")
    run_with_ngrok(app)


############################################################
# Export Model読込
############################################################
# signature読込
sig_holder = export_holder + 'signature.json'
with open(sig_holder, 'r') as f:
    signature = json.load(f)

inputs = signature.get('inputs')
outputs = signature.get('outputs')

# モデルの学習画像サイズを取得
input_width, input_height = inputs['Image']['shape'][1:3]
# print("input_width={0}, input_height={1}".format(input_width, input_height))

# model読込
model = tf.saved_model.load(export_holder)
infer = model.signatures['serving_default']


############################################################
# 初期設定
############################################################
# ラベル
label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


############################################################
# index.html
############################################################
@app.route("/")
def index():
    return render_template('index.html')


############################################################
# Input.jsからPOSTされたときに動作
############################################################
@app.route('/output', methods=['POST'])
def output():
    # json形式でデータを受け取る
    b64_pngdata = request.json['b64_pngdata']
#   display(b64_pngdata, "b64_pngdata")

    # base64デコード
    tmpdata = b64_pngdata.split(',')  # base64のヘッダを削除
    bindata = base64.b64decode(tmpdata[1])  # 「data:image/png;base64,～」以降のデータのみをデコード
#   display(bindata, "bindata")

    # pillow形式で読み込み画像のリサイズを行う
    imgPIL = Image.open(BytesIO(bindata)).convert('RGB')
#   imgPIL = imgPIL.convert('L')                                   # グレースケール
    imgPIL = imgPIL.resize((input_width, input_height))            # リサイズ(224 x 224)
#   imgPIL.show()

    # pillow形式→バイナリ変換
    with BytesIO() as output_png:
        imgPIL.save(output_png, format="PNG")
        contents = output_png.getvalue()  # バイナリ取得

    # mnist形式
    x = img_to_array(imgPIL) / 255
    x = x[None, ...]

    # shape確認
#   print(x.shape)
#   print("")

    # 遊星からの物体Xの推論
    # 推論
    predict = infer(tf.constant(x))
#   print(predict)
#   print()

    # 推論結果
    predict = infer(tf.constant(x))['Prediction'][0]
    print('推測結果：', predict.numpy().decode())

    # 確率
    confidence = infer(tf.constant(x))['Confidences'][0]
    confidence = confidence.numpy()
    print('確率    ： {:1.010f}'.format(confidence[int(predict)]))
    print()

    for i, conf in enumerate(confidence):
        print(i, '{:1.010f}'.format(conf))
    print()

    # 推測ラベル
    pred_label = label[int(predict)]
#   print("label:", pred_label)

    # 推測スコア
#   score = str("{:.10f}".format(np.max(confidence)))
    score = str("{:.10f}".format(confidence[int(predict)]))
#   print("score:", score)
#   print("------------------------------------------------------------------")

    # base64エンコード
    tmpdata = str(base64.b64encode(contents))
#   display(tmpdata, "tmpdata")
    tmpdata = tmpdata[2:-1]  # 「b’～’」の中身だけをエンコード

    data1 = "data:image/png;base64," + tmpdata
    data2 = pred_label
    data3 = score
    label_score = [str("{:.10f}".format(n)) for n in confidence]
#   print(label_name)

    return_data = {"pred_png": data1,
                   "pred_label": data2,
                   "pred_score": data3,
                   "label0": label_score[0],
                   "label1": label_score[1],
                   "label2": label_score[2],
                   "label3": label_score[3],
                   "label4": label_score[4],
                   "label5": label_score[5],
                   "label6": label_score[6],
                   "label7": label_score[7],
                   "label8": label_score[8],
                   "label9": label_score[9]
                   }

    return jsonify(ResultSet=json.dumps(return_data))


############################################################
# debug用
############################################################
def display(data, name):
    print(name)
    print(data)
    print(type(data))
    print("")


############################################################
# flask起動
############################################################
if __name__ == '__main__':
    app.debug = False
    app.run()
