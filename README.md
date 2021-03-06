# mnister_for_tensorflow
Perform MNIST using Lobe(Export Tensorflow Model).
![system](https://user-images.githubusercontent.com/69660581/100043780-75e36e80-2e51-11eb-8f06-3906b9b4341a.png)

## ■Requirements
- [Lobe](https://lobe.ai/) -> export TensorFlow model. ( Please prepare your own Export Tensorflow Model. )
- Server：Local PC or Google Colab
- Browser：Local PC

## ■Starting Method
- Start the server on your local PC or Google Colab.
- Start Browser on your local PC and access the server ( http://127.0.0.1:5000/ ).

※See below for details. ( https://qiita.com/PoodleMaster/items/af5630305df52bed0564 )

### ■When starting the server on the local PC
![sysytem](https://user-images.githubusercontent.com/69660581/100050006-eb543c80-2e5b-11eb-8bdb-bbe7599a8335.png)

#### (1)Start the server
- Initial settings for starting the server.
``` 
git clone https://github.com/PoodleMaster/mnister_for_Tensorflow
cd mnister_for_Tensorflow
conda create -n mnister_tf python=3.7
conda activate mnister_tf
pip install tensorflow pillow flask flask_ngrok
```
- Upload `export model`[(※1)](https://github.com/PoodleMaster/mnister_for_tensorflow/blob/main/README.md#supplement) to` mnister_for_tensorflow` folder

※Like this.
```
mnister_for_Tensorflow
  │
  ├─export model
  │
```
- Start the server.
```
python server.py
```

#### (2)Access the server
- Access "http://127.0.0.1:5000/" with a browser.


### ■When starting the server on the Google Colab
![system](https://user-images.githubusercontent.com/69660581/100050002-eabba600-2e5b-11eb-87f5-8583f43e31f9.png)
sample:https://github.com/PoodleMaster/mnister_for_tensorflow/blob/main/mnister_for_tensorflow.ipynb

#### (1)Start the server
- Initial settings for starting the server.
``` 
!git clone https://github.com/PoodleMaster/mnister_for_Tensorflow
%cd mnister_for_Tensorflow
!pip install flask flask_ngrok
```
- Upload `export model`[(※1)](https://github.com/PoodleMaster/mnister_for_tensorflow/blob/main/README.md#supplement) to` mnister_for_tensorflow` folder

※Like this.
```
mnister_for_Tensorflow
  │
  ├─export model
  │
```
- Start the server.
```
!python server.py --colab
```

#### (2)Access the server
- Access the URL (http://xxxxxxxxxxxx.ngrok.io) issued by NGROK.

## ■Supplement
※1：export model <- Tensorflow Model exported by Lobe.
```
export model
   │
   ├─example
   │ │
   │ ├─README.md
   │ ├─requirements.txt
   │ └─tf_example.py
   │
   ├─variables
   │ │
   │ ├─variables.data-00000-of-00001
   │ └─variables.index
   │
   ├─saved_model.pb
   └─signature.json
```

## ■Contributing
Contributions, issues and feature requests are welcome.

## ■Author
Github: PoodleMaster

## ■License
Check the LICENSE file.
