import cv2
import numpy as np
import pyautogui
import math

def visualize_roi(num_hands_detect, centre, score_thresh, scores, rad_in, rad_mid, rad_out, image_np):
    for i in range(num_hands_detect):
        if(scores[i] > score_thresh):
            cv2.rectangle(image_np, (0, 0), (image_np.shape[1], image_np.shape[0]), (0, 255, 0), -1)
            cv2.circle(image_np, centre, rad_out, (255, 0, 0), -1)
            cv2.circle(image_np, centre, rad_mid, (0, 255, 0), -1)
            cv2.circle(image_np, centre, rad_in, (255, 0, 0), -1)


def visualize_blurred(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if(scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (0, 0)
            p2 = (int(left), int(im_height))
            p3 = (int(left), int(bottom))
            p4 = (int(im_width), int(im_height))
            p5 = (int(right), 0)
            p6 = (int(im_width), int(top))
            # p1 = (int(left), int(top))
            # p2 = (int(right), int(bottom))

            cv2.rectangle(image_np, p1, p2, (192,192,192), -1)
            cv2.rectangle(image_np, p3, p4, (192,192,192), -1)
            cv2.rectangle(image_np, p5, p4, (192,192,192), -1)
            cv2.rectangle(image_np, p1, p6, (192,192,192), -1)
# 
            # print(type(left))
            # sub_img = image_np[int(0): int(left)]
            # white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            # 
            # res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
# 
            # image_np[0: left, None] = res
            # return image_np

def draw_trajectory(X, Y, image_np):
    px, py = -1, -1
    for i in range(len(X)):
        if px < 0:
            px, py = X[i], Y[i]
            continue
        cv2.line(image_np, (px, py), (X[i], Y[i]), (0,255,0), 1)
        px, py = X[i],Y[i]

def dist(x, y, px, py):
    return ((x-px)**2 + (y-py)**2) ** 0.5

def get_direction(x, y, px, py):
    # n,w,s,e -> 0,1,2,3
    cx, cy = 3, 0
    yy, xx = py-y, x-px

    if yy<0 and xx<0:
        cx, cy=1,2
    elif yy<0:
        cy=2
    elif xx<0:
        cx=1
    deg = -1

    if xx == 0:
        deg = 90.0
    else:
        deg = math.degrees(math.atan(abs(yy) / abs(xx)))

    # print("yy, xx: ", yy, xx)
    # print("deg: ", deg)
    if deg<=45:
        return cx
    return cy

def probab(p):
    return np.max(p) / np.sum(p)

def control_mouse_pointer(x, y, screenWd, screenHt, pred, prev_pred):
    if pred == 0 or pred == 4:
        currx, curry = pyautogui.position()

        currx += x
        curry += y

        currx = max(1, currx)
        currx = min(screenWd-1, currx)

        curry = max(1, curry)
        curry = min(screenHt-1, curry)

        pyautogui.moveTo(currx,curry)

    elif pred == 1 and not(pred == prev_pred):
        pyautogui.click()

def control_vlc(swipe, pred, state):
    if state < 0 and (pred == 1 or pred == 3):
        # print(pyautogui.getWindows())
        # l,t,w,h = pyautogui.locateOnScreen('vlc2.png')
        # moveToX = int(l + w/2)
        # moveToY = int(t + h/2)
        # print(moveToX, moveToY)
        # pyautogui.moveTo(moveToX, moveToY)
        pyautogui.click()
        pyautogui.press('space')
        state = 1

        if pred == 1:
            state = 0
        print("state", state)

    if state == 1 and pred == 1:
        pyautogui.press('space')
        state = 0

    elif state == 0 and pred == 3:
        pyautogui.press('space')
        state = 1

    elif pred == 0:
        if swipe == 1:
            pyautogui.hotkey('alt', 'left')

        elif swipe == 3:
            pyautogui.hotkey('alt', 'right')

    elif pred == 4:
        if swipe == 0:
            pyautogui.hotkey('ctrl', 'up')
            pyautogui.hotkey('ctrl', 'up')

        elif swipe == 2:
            pyautogui.hotkey('ctrl', 'down')
            pyautogui.hotkey('ctrl', 'down')

    return state
