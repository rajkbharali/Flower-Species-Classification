import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import os
import cv2
import mahotas as mt
from sklearn import metrics

path='dataset/data'
path2='dataset/test_color6'
fixed_size = tuple((500, 500))

global_features=[]
labels=[]

def extract_color(im2):
    histr=cv2.calcHist([im2],[0,1,2],None,[8,8,8],[0,255,0,255,0,255])
    return histr.flatten()

def extract_texture(im2):
    texture=mt.features.haralick(im2)
    ht_mean=texture.mean(axis=0)
    return ht_mean.flatten()

def extract_shape(im2):
    shape=cv2.HuMoments(cv2.moments(im2))
    return shape.flatten()
i=0
for fol in os.listdir(path): #feature extraction
    new_path=path+'/'+fol
    i=i+1
    cur_dir=fol
    for image in os.listdir(new_path):
        des=new_path+'/'+image
        im = cv2.imread(des)
        img = cv2.imread(des)
        if im is not None:
            im = cv2.resize(im, fixed_size)
            feature1=extract_color(im) #color
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            feature2=extract_texture(gray) #texture
            feature3=extract_shape(gray) #shape
            global_feature = np.hstack([feature1, feature2, feature3])
            labels.append(cur_dir)
            global_features.append(global_feature)
        else:
            print('Failed to open the file')
    print ("[STATUS] processed folder: {}.{}".format(i,cur_dir))

global_feature=np.reshape(global_feature,(-1,532))
Tr_global_feature=np.transpose(global_features)
Tr2_global_feature=np.transpose(Tr_global_feature)

new_rescaled_features=np.array(Tr2_global_feature)
new_labels=np.array(labels)

import warnings
warnings.filterwarnings('ignore')


font=cv2.FONT_HERSHEY_SIMPLEX
pos=(10,50)
fontScale=2
fontColor=(255,255,255)
lineType=2

i=0
print('----Testing [STATUS]----')
for fol in os.listdir(path2):
    new_path=path2+'/'+fol
    i=i+1
    cur_dir=fol
    for image in os.listdir(new_path):
        des=new_path+'/'+image
        im = cv2.imread(des)
        img = cv2.imread(des)
        if im is not None:
            feature4=extract_color(im)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            feature5=extract_texture(gray)
            feature6=extract_shape(gray)
            global_feature = np.hstack([feature4, feature5, feature6])
            new_feature=np.array(global_feature)
            main_feature=np.reshape(new_feature,(-1,532))
            classifier=RandomForestClassifier(n_estimators=100)  
            classifier.fit(new_rescaled_features,new_labels)
            y_pred=classifier.predict(main_feature)
            #print("\nClass:",y_pred)
            text=str(y_pred)
            cv2.putText(im,text,pos,font,fontScale,fontColor,lineType)
            plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            plt.show()
        else:
            print('Failed to open the file')
    print ("[STATUS] processed folder: {}.{}".format(i,cur_dir))

#print("Accuracy:",metrics.accuracy_score(test_labels,y_pred))