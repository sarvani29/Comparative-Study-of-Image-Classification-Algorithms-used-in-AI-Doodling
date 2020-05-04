#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from collections import deque
import os


# In[ ]:


from keras.models import load_model
from keras.utils import print_summary


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


cnn_checkpoint_path = "D:\Lectures\sem 2\Artificial Intelligence\CS7IS2-project\Kanika\cnn_model.h5"
emojis_path = 'D:\Lectures\sem 2\Artificial Intelligence\CS7IS2-project\Kanika\emoji\\'


# In[ ]:


model= load_model(cnn_checkpoint_path)


# In[ ]:


print_summary(model)


# In[ ]:


def process_image(image):
  x=28
  y=28
  image = cv2.resize(image, (x, y))
  image = np.array(image, dtype=np.float32)
  image = np.reshape(image, (-1, x, y, 1))
  print (image)
  return image


# In[ ]:


def predict(model, image):
  processed_image = process_image(image)
  print ("processed image shape: ")
  print (processed_image.shape)
  prediction_temp = model.predict(processed_image)
  print ("prediction temp: ")
  print (prediction_temp)
  prediction = model.predict(processed_image)[0]
  print ("prediction: ")
  print (prediction)
  prediction_class = list(prediction).index(max(prediction))
  return max(prediction), prediction_class


# In[ ]:


def get_emoji():
#   path = '/content/drive/My Drive/Colab Notebooks/CS7IS2 - Artificial Intelligence/emoji'
  emojis = []
  for e in range(len(os.listdir(emojis_path))):
    print (e)
    emojis.append(cv2.imread(emojis_path + str(e) + '.png', -1))
  return emojis


# In[ ]:


def blend_transparent(face_img, overlay_t_img):
  # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


# In[ ]:


def overlay(img, emoji, x, y, w, h):
  emoji = cv2.resize(emoji, (w,h))
  try:
    img[y:y+h, x:x+w] = blend_transparent(img[y:y+h, x:x+w], emoji)
  except:
    pass
  return img


# In[ ]:


predict(model, np.zeros((50, 50, 1), dtype=np.uint8))


# In[ ]:


# emojis = get_emoji()
# # emojis


# In[ ]:


# Lower_green = np.array([110, 50, 50])
# Upper_green = np.array([130, 255, 255])
# points = deque(maxlen=512)
# blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
# digit = np.zeros((200, 200, 3), dtype=np.uint8)
# pred_class = 0


# In[ ]:


# capture = cv2.VideoCapture(0)
# if(capture.isOpened() == False):
#   print("not open")
# else:
#   print ("open")


# In[ ]:


def main():
    emojis = get_emoji()
#     print ("emojis: ")
#     print (emojis)
    cap = cv2.VideoCapture(0)
    Lower_green = np.array([110, 50, 50])
    Upper_green = np.array([130, 255, 255])
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    pred_class = 0

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
#         cv2.imshow("frame1", img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         cv2.imshow("frame2", hsv)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, Lower_green, Upper_green)
#         cv2.imshow("frame3", mask)
        mask = cv2.erode(mask, kernel, iterations=2)
#         cv2.imshow("frame4", mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         cv2.imshow("frame5", mask)
        # mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
#         cv2.imshow("frame6", mask)
        res = cv2.bitwise_and(img, img, mask=mask)
#         cv2.imshow("res", res)
        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None
        print ("cnts: " + str(cnts))
        print ("heir: " + str(heir))

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 200:
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)

                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                    cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
        elif len(cnts) == 0:
            if len(pts) != []:
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#                 blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                blackboard_cnts, h = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(blackboard_cnts) >= 1:
                    cnt = max(blackboard_cnts, key=cv2.contourArea)
                    print(cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) > 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        digit = blackboard_gray[y:y + h, x:x + w]
                        pred_probab, pred_class = predict(model, digit)
#                         print(pred_class, pred_probab)

            pts = deque(maxlen=512)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            img = overlay(img, emojis[pred_class], 400, 250, 100, 100)
        cv2.imshow("Frame", img)
        k = cv2.waitKey(10)
        if k == 27:
            break
            
main()


# In[ ]:


cv2.__version__

