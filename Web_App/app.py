from flask import Flask, render_template, request,jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re
import torch
import torchvision
from torch import nn
from torchvision import models
from torchvision import transforms


device = torch.device('cpu')

class Net(nn.Module):

	def __init__(self, model):

		super(Net, self).__init__()

		self.resnet = nn.Sequential(*list(model.children())[:-1])

		self.fc = nn.Linear(in_features= 2048, out_features=2)


	def forward(self, x):

		x = self.resnet(x)

		x = x.view(x.shape[0], -1)

		x = self.fc(x)

		return x








img_size=100

from torchvision import models
resnet152 = models.resnet152(pretrained=False)
net = Net(resnet152)

app = Flask(__name__) 

model= torch.load(r'model.pkl', map_location='cpu')

label_dict={0:'Maligant', 1:'Bengin'}


def preprocess2(img):


	T1 = transforms.Resize(256) # 随机裁剪 

	img = T1(img)


	T2 =transforms.CenterCrop(224)

	img = T2(img)



	T3 = transforms.ToTensor()

	img = T3(img)



	T4 = transforms.Normalize([0.485, 0.456, 0.406],

	                          [0.229, 0.224, 0.225])
	img = T4(img)



	img = torch.unsqueeze(img, 0)

	print(img ,' unsqueeze')

	model.eval()

	print(' done evalution')
	img = img.to(device)

	return img

	




def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=resized.reshape(1,img_size,img_size)
	return reshaped

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	img=preprocess2(image)




	outputs = model(img)


	_, preds = torch.max(outputs, 1)

	result = torch.max(outputs)

	preds = preds.cpu().numpy()

	preds = preds[0]


	if(preds==0):

	    label = 'Benign Tumor'
	    
	else:
	    
	    label = 'Malignant Tumor'


	# prediction = model(test_image)
	# print(prediction)
	# result=np.argmax(prediction,axis=1)[0]
	# accuracy=float(np.max(prediction,axis=1)[0])

	# label=label_dict[result]

	# print(prediction,result,accuracy)

	response = {'prediction': {'result': label}}

	return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">


# img = Image.open(r'/content/drive/MyDrive/Datasets/61/test/205.png')

# plt.subplot(1,1,1)

# plt.imshow(img) 

# plt.show() 

T1 = transforms.Resize(256) # 随机裁剪 

img = T1(img)

T2 =transforms.CenterCrop(224)

img = T2(img)

T3 = transforms.ToTensor()

img = T3(img)

T4 = transforms.Normalize([0.485, 0.456, 0.406],

                          [0.229, 0.224, 0.225])
img = T4(img)

img = img.unsqueeze(0)

model.eval()

img = img.to(device)

outputs = model(img)

# # print(outputs)

_, preds = torch.max(outputs, 1)

result = torch.max(outputs)

print('The output is:')

print(outputs)

preds = preds.cpu().numpy()

preds = preds[0]

if(preds==0):

    print('non breast cancer')
    
else:
    
    print('breast cancer')