import pandas as pd
import tensorflow
from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1./255.,rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2)

data = pd.read_csv('/content/drive/My Drive/dance dataset/train.csv')

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

data['Path'] = "/content/drive/My Drive/dance dataset/train/"+data['Image']

main_data = data.groupby("target")
main_data = main_data.apply(lambda x : x.sample(main_data.size().min()).reset_index(drop=True))

val_dat = pd.read_csv("/content/drive/My Drive/dance dataset/test_new1.csv")

val_dat['path'] = "/content/drive/My Drive/dance dataset/test/"+val_dat['Image']

trdata = ImageDataGenerator(rescale=1./255.,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
            
                            )
valdata = ImageDataGenerator(rescale=1./255.)

traindata = trdata.flow_from_dataframe(data,x_col='Path',y_col='target',target_size=(224,224),shuffle=False,class_mode='categorical')

validdata = valdata.flow_from_dataframe(val_dat,x_col='path',y_col='Target',target_size=(224,224),shuffle=False,class_mode='categorical')

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten,Activation,AveragePooling2D,GlobalAveragePooling2D
import pandas as pd
from keras.models import Sequential,Model
import keras.backend as K
from keras import regularizers
import numpy as np

vgg = VGG16(weights='imagenet',input_shape=(224,224,3),include_top=False)
for layer in vgg.layers[:15]:
  layer.trainable = False
resn = ResNet50(weights='imagenet',input_shape=(224,224,3),include_top=False)
for layer in resn.layers[:15]:
  layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(l=0.00001))(x)
drp1 = Dropout(0.2)(x)
x = Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(l=0.00001))(drp1)
drp2 = Dropout(0.2)(x)
prediction = Dense(8, activation='softmax')(drp2)
model = Model(vgg.input,output = prediction)
x = Flatten()(resn.output)
x = Dense(1024,activation='relu',kernel_regularizer=regularizers.l2(l=0.00001))(x)
prediction = Dense(8, activation='softmax')(x)
model_res = Model(resn.input,output = prediction)

model.summary()

from keras.optimizers import Adam,RMSprop,SGD
opt = SGD(learning_rate=0.0001)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy',f1_m])

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback,History,ModelCheckpoint,EarlyStopping
callbacker = EarlyStopping(monitor='val_f1_m',verbose=1,patience=20,mode='max')
checker = ModelCheckpoint(filepath="dance_res.h5",monitor="val_f1_m",verbose=1,save_best_only=True,mode='max',period=1)

model.fit_generator(traindata,steps_per_epoch=11,epochs=150,validation_data=validdata,validation_steps=11,callbacks=[callbacker,checker])

pred = pd.read_csv("/content/drive/My Drive/dance dataset/test.csv")

pred['Path'] = "/content/drive/My Drive/dance dataset/test/"+data['Image']

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
dataframe=pred,
directory="/content/drive/My Drive/dance dataset/test/",
x_col="Image",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(224,224))

from keras.models import load_model
from keras.engine.training_generator import evaluate_generator,predict_generator
import pandas as pd
dependency = {
    'f1_m': f1_m
}
mod = load_model("/content/dance.h5",custom_objects=dependency)

predict = mod.predict(test_generator)

evaluate_generator(mod,validdata,steps=11)

TEST = np.argmax(predict,axis=1)
print(TEST)
y = TEST

validdata.class_indices

Y=[]

for j in range(0,156):
  Y.append(0)
for i in range(0,156):
  if(y[i]== 4):
    Y[i] = 'manipuri'
  if(y[i]==0):
    Y[i] = 'bharathanatyam'
  if(y[i]==6):
    Y[i] = 'odissi'
  if(y[i]==2):
    Y[i] = 'kathakali'
  if(y[i]==1):
    Y[i] = 'kathak'
  if(y[i]==7):
    Y[i] = 'sattriya'
  if(y[i]==3):
    Y[i] = 'kuchipudi'
  if(y[i]==5):
    Y[i] = 'mohiniyattam'
Y = np.array(Y)

print(Y)

read = pd.read_csv("/content/drive/My Drive/dance dataset/test.csv")
read['target'] = Y
read.to_csv('submission.csv',index = False)

mode = ResNet50(weights='imagenet')
predicc = mode.predict(test_generator)

from keras.applications.resnet50 import decode_predictions
decode = decode_predictions(predicc,top=1)

print(decode)

