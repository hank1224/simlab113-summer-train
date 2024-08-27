# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:05:28 2023

@author: user
"""

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils # 設定繪畫手部節點及線條的資訊
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
# IMAGE_FILES = []
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#   for idx, file in enumerate(IMAGE_FILES):
#     # Read an image, flip it around y-axis for correct handedness output (see
#     # above).
#     image = cv2.flip(cv2.imread(file), 1)
#     # Convert the BGR image to RGB before processing.
#     results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print handedness and draw hand landmarks on the image.
#     print('Handedness:', results.multi_handedness)
#     if not results.multi_hand_landmarks:
#       continue
#     image_height, image_width, _ = image.shape
#     annotated_image = image.copy()
#     for hand_landmarks in results.multi_hand_landmarks:
#       print('hand_landmarks:', hand_landmarks)
#       print(
#           f'Index finger tip coordinates: (',
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#           f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#       )
#       mp_drawing.draw_landmarks(
#           annotated_image,
#           hand_landmarks,
#           mp_hands.HAND_CONNECTIONS,
#           mp_drawing_styles.get_default_hand_landmarks_style(),
#           mp_drawing_styles.get_default_hand_connections_style())
#     cv2.imwrite(
#         '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#     # Draw hand world landmarks.
#     if not results.multi_hand_world_landmarks:
#       continue
#     for hand_world_landmarks in results.multi_hand_world_landmarks:
#       mp_drawing.plot_landmarks(
#         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5, # 設定手部檢測時的準確程度，只要符合程度達到50%即啟動
    min_tracking_confidence=0.5) as hands: # 抓到的手部影像符合程度，影響演算法是否重新啟動手部檢測功能，只要符合程度達到50%即忽略手部檢測功能
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 將 BGR 轉換成 RGB，因為檢測算法輸入影像的顏色格式順序需要RGB，而透過OpenCV讀取的影像顏色順序為BGR，所以需要透過程式做轉換，才能將轉換後的影像丟到檢測算法裡取得結果。
    results = hands.process(image) # 偵測手掌

    # Draw the hand annotations on the image. 將節點和骨架繪製到影像中
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

     #  print("thumb=",hand_landmarks.landmark[4].x, hand_landmarks[4].y, hand_landmarks.landmark[3].x,hand_landmarks.landmark[3].y)
       if hand_landmarks.landmark[8].y<hand_landmarks.landmark[7].y:
           cv2.putText(image,"Pointing up", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        # Initially set finger count to 0 for each cap
       fingerCount = 0
       if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            # Get hand index to check label (left or right)
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label

            # Set variable to keep landmarks positions (x and y)
            handLandmarks = []

            # Fill list with x and y positions of each landmark
            for landmarks in hand_landmarks.landmark:
              handLandmarks.append([landmarks.x, landmarks.y])

            # Test conditions for each finger: Count is increased if finger is considered raised.
            # Thumb: TIP x position must be greater or lower than IP x position, deppeding on hand label.
            if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
              fingerCount = fingerCount+1
            elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
              fingerCount = fingerCount+1

            # Other fingers: TIP y position must be lower than PIP y position,
            #   as image origin is in the upper left corner.
            if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger
              fingerCount = fingerCount+1
            if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
              fingerCount = fingerCount+1
            if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
              fingerCount = fingerCount+1
            if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
              fingerCount = fingerCount+1

        # Draw hand landmarks 將結果標註在影像上，填入要標註的影像、每個結果點、標註方法。依照對應的是繪製標註點還是標註線
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

          # Display finger count
            cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
        # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:#按esc關閉
      break


cap.release()
cv2.destroyAllWindows()
