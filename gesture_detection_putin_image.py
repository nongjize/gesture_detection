import numpy as np
import cv2
from keras.models import load_model
import sys

threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
prediction = ''
action = ''
score = 0


gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

model = load_model('VGG_cross_validated.h5')


def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print('pred_array:')
    print(pred_array)
    result = gesture_names[np.argmax(pred_array)]
    print('Result:')
    print(result)
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score

def load_image_show(path):
    #图片加载
    image=cv2.imread(path)

    #图片处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    # cv2.imshow('blur', blur)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # copies 1 channel BW image to all 3 RGB channels
    target = np.stack((thresh,) * 3, axis=-1)
    target = cv2.resize(target, (224, 224))    
    target = target.reshape(1, 224, 224, 3)

    #输入模型predict
    prediction, score = predict_rgb_image_vgg(target)
      
    #显示图片+模型预测结果
    showimage=gray;
    cv2.putText(showimage, "Prediction: {0} ({1}%)".format(prediction,score), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0))
    cv2.imshow("press any key to close--{0} ({1}%)".format(prediction,score),showimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

while (1):
    image_name=input("请输入图片文件(输入exit退出):\n")
    if image_name == "exit":  # press ESC to exit all windows at any time
        break
    else:
        load_image_show(image_name)




