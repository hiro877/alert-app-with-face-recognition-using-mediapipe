#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from utils import AlarmThread

import sys
from plyer import notification
import tkinter as tk
from tkinter import messagebox
import tkinter.simpledialog as simpledialog
import statistics
import time
import csv

import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_faces", type=int, default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args

def do_alert(alert_title, alert_message):
    # alert = "Let's move the body.\nYou have been concentrating on your work for more than 30 minutes."
    notification.notify(
        title = alert_title,
        message = alert_message,
        app_name = "モニター監視"
        )

def enqueue(ndarray, data):
    if (type(ndarray) is list):
        ndarray = np.array(ndarray)
    queue = ndarray[1:]
    queue = np.append(queue, data)
    return queue

def main():
    # my add variable value ################################################
    # is_concentrated = False
    # frame_length       = 256
    sampling_time      = 0.1
    detecting_time     = 0
    not_detecting_time = 0
    window_size        = 100
    list_zeros_window  = [0 for i in range(window_size)]
    list_detect_window = list_zeros_window
    detect_count       = 0
    border_face_detect = 0.1
    # alert_time = 60 * 30
    alert_time = 20
    alert_title = "Alert"
    alert_message = "Let's move the body.\nYou have been concentrating on your work for more than 25 minutes."

    # Reset Param ############################################################
    # reset_work_boarder = 60 * 3
    reset_work_boarder = 3

    # Exercise ###############################################################
    # exercise_boarder = 60*5
    exercise_boarder = 5
    is_exercise_state = False
    is_init_exercise_state = True
    finish_alert_title = "Congratulation！"
    finish_alert_message= "Finished Your Exercise Time."

    # Repeat Alert ###########################################################
    repeat_alert_time = 1
    # repeat_boarder = 60
    repeat_boarder = 10
    repeat_alert_title = "Repeat Alerm"
    repeat_alert_message = "Let's move the body.\n"

    # investigation param ####################################################
    output_list = [['time[s]', '顔検出値[0 or 1]']]
    output_count = 0

    # plot param #############################################################
    x = np.linspace(0, 10, 100)
    y = np.zeros(x.size)
    plt.ion()
    figure, ax = plt.subplots(figsize=(8,6))
    line1, = ax.plot(x, y)
    plt.title("Dynamic Plot of face detection",fontsize=25)
    plt.xlabel("time",fontsize=18)
    plt.ylabel("is detected value",fontsize=18)
    plt.ylim(-1.1,1.1)
    updated_y = y
    plot_data = 0
    # border prot
    yb = np.full((1,100), border_face_detect)[0]
    plt.plot(x, yb, color = 'red')


    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_faces = args.max_num_faces
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 描画 ################################################################
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, face_landmarks)
                # 描画
                debug_image = draw_landmarks(debug_image, face_landmarks)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            list_detect_window[detect_count]=1
            output_list.append([output_count, 1])
            plot_data=1
        else:
            list_detect_window[detect_count]=0
            output_list.append([output_count, 0])
            plot_data=0

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MediaPipe Face Mesh Demo', debug_image)

        # calculate the mean of list_detect_window ############################
        mean_list_detect_window = statistics.mean(list_detect_window)
        if mean_list_detect_window > border_face_detect:
            detecting_time += sampling_time
            not_detecting_time = 0
            if detecting_time >= alert_time:
                is_exercise_state = True
                is_init_exercise_state = True
                if repeat_alert_time % repeat_boarder == 0:
                    print("repeat alert")
                    do_alert(repeat_alert_title, repeat_alert_message)
                elif repeat_alert_time ==1:
                    do_alert(alert_title, alert_message)
                repeat_alert_time = round(repeat_alert_time + sampling_time, 1)
                print(repeat_alert_time)
        else:
            if is_exercise_state:
                # if is_init_exercise_state:
                #     mean_list_detect_window = list_zeros_window
                #     is_init_exercise_state = False

                not_detecting_time += sampling_time
                if not_detecting_time >= exercise_boarder:
                    do_alert(finish_alert_title, finish_alert_message)
                    # Reset Param ################################################
                    detecting_time = 0
                    not_detecting_time=0
                    repeat_alert_time=1
                    is_exercise_state = False
            else:
                ### Personal function ########################################
                ### 作業していない期間が一定時間を超えたら作業タイマーをリセットする
                ##############################################################
                not_detecting_time += sampling_time
                if not_detecting_time >= reset_work_boarder:
                    # Reset Param ############################################
                    detecting_time = 0
                    not_detecting_time=0
                    repeat_alert_time=1
        
        if is_exercise_state and is_init_exercise_state:
            # print("test test test")
            list_detect_window = list_zeros_window
            is_init_exercise_state = False

        

        detect_count = 0 if detect_count == window_size-1 else detect_count+1
        time.sleep(sampling_time)
        output_count += sampling_time

        # Update Plot ########################################################
        # updated_y = enqueue(updated_y, plot_data) 
        updated_y = enqueue(updated_y, mean_list_detect_window)
        line1.set_xdata(x)
        line1.set_ydata(updated_y)
        figure.canvas.draw()
        figure.canvas.flush_events()

    cap.release()
    cv.destroyAllWindows()
    export_list_csv(output_list, 'face_detected.csv')




def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x, landmark_y))

        cv.circle(image, (landmark_x, landmark_y), 1, (0, 255, 0), 1)

    if len(landmark_point) > 0:
        # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

        # 左眉毛(55：内側、46：外側)
        cv.line(image, landmark_point[55], landmark_point[65], (0, 255, 0), 2)
        cv.line(image, landmark_point[65], landmark_point[52], (0, 255, 0), 2)
        cv.line(image, landmark_point[52], landmark_point[53], (0, 255, 0), 2)
        cv.line(image, landmark_point[53], landmark_point[46], (0, 255, 0), 2)

        # 右眉毛(285：内側、276：外側)
        cv.line(image, landmark_point[285], landmark_point[295], (0, 255, 0),
                2)
        cv.line(image, landmark_point[295], landmark_point[282], (0, 255, 0),
                2)
        cv.line(image, landmark_point[282], landmark_point[283], (0, 255, 0),
                2)
        cv.line(image, landmark_point[283], landmark_point[276], (0, 255, 0),
                2)

        # 左目 (133：目頭、246：目尻)
        cv.line(image, landmark_point[133], landmark_point[173], (0, 255, 0),
                2)
        cv.line(image, landmark_point[173], landmark_point[157], (0, 255, 0),
                2)
        cv.line(image, landmark_point[157], landmark_point[158], (0, 255, 0),
                2)
        cv.line(image, landmark_point[158], landmark_point[159], (0, 255, 0),
                2)
        cv.line(image, landmark_point[159], landmark_point[160], (0, 255, 0),
                2)
        cv.line(image, landmark_point[160], landmark_point[161], (0, 255, 0),
                2)
        cv.line(image, landmark_point[161], landmark_point[246], (0, 255, 0),
                2)

        cv.line(image, landmark_point[246], landmark_point[163], (0, 255, 0),
                2)
        cv.line(image, landmark_point[163], landmark_point[144], (0, 255, 0),
                2)
        cv.line(image, landmark_point[144], landmark_point[145], (0, 255, 0),
                2)
        cv.line(image, landmark_point[145], landmark_point[153], (0, 255, 0),
                2)
        cv.line(image, landmark_point[153], landmark_point[154], (0, 255, 0),
                2)
        cv.line(image, landmark_point[154], landmark_point[155], (0, 255, 0),
                2)
        cv.line(image, landmark_point[155], landmark_point[133], (0, 255, 0),
                2)

        # 右目 (362：目頭、466：目尻)
        cv.line(image, landmark_point[362], landmark_point[398], (0, 255, 0),
                2)
        cv.line(image, landmark_point[398], landmark_point[384], (0, 255, 0),
                2)
        cv.line(image, landmark_point[384], landmark_point[385], (0, 255, 0),
                2)
        cv.line(image, landmark_point[385], landmark_point[386], (0, 255, 0),
                2)
        cv.line(image, landmark_point[386], landmark_point[387], (0, 255, 0),
                2)
        cv.line(image, landmark_point[387], landmark_point[388], (0, 255, 0),
                2)
        cv.line(image, landmark_point[388], landmark_point[466], (0, 255, 0),
                2)

        cv.line(image, landmark_point[466], landmark_point[390], (0, 255, 0),
                2)
        cv.line(image, landmark_point[390], landmark_point[373], (0, 255, 0),
                2)
        cv.line(image, landmark_point[373], landmark_point[374], (0, 255, 0),
                2)
        cv.line(image, landmark_point[374], landmark_point[380], (0, 255, 0),
                2)
        cv.line(image, landmark_point[380], landmark_point[381], (0, 255, 0),
                2)
        cv.line(image, landmark_point[381], landmark_point[382], (0, 255, 0),
                2)
        cv.line(image, landmark_point[382], landmark_point[362], (0, 255, 0),
                2)

        # 口 (308：右端、78：左端)
        cv.line(image, landmark_point[308], landmark_point[415], (0, 255, 0),
                2)
        cv.line(image, landmark_point[415], landmark_point[310], (0, 255, 0),
                2)
        cv.line(image, landmark_point[310], landmark_point[311], (0, 255, 0),
                2)
        cv.line(image, landmark_point[311], landmark_point[312], (0, 255, 0),
                2)
        cv.line(image, landmark_point[312], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[82], (0, 255, 0), 2)
        cv.line(image, landmark_point[82], landmark_point[81], (0, 255, 0), 2)
        cv.line(image, landmark_point[81], landmark_point[80], (0, 255, 0), 2)
        cv.line(image, landmark_point[80], landmark_point[191], (0, 255, 0), 2)
        cv.line(image, landmark_point[191], landmark_point[78], (0, 255, 0), 2)

        cv.line(image, landmark_point[78], landmark_point[95], (0, 255, 0), 2)
        cv.line(image, landmark_point[95], landmark_point[88], (0, 255, 0), 2)
        cv.line(image, landmark_point[88], landmark_point[178], (0, 255, 0), 2)
        cv.line(image, landmark_point[178], landmark_point[87], (0, 255, 0), 2)
        cv.line(image, landmark_point[87], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[317], (0, 255, 0), 2)
        cv.line(image, landmark_point[317], landmark_point[402], (0, 255, 0),
                2)
        cv.line(image, landmark_point[402], landmark_point[318], (0, 255, 0),
                2)
        cv.line(image, landmark_point[318], landmark_point[324], (0, 255, 0),
                2)
        cv.line(image, landmark_point[324], landmark_point[308], (0, 255, 0),
                2)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image

def export_list_csv(export_list, csv_dir):

    with open(csv_dir, "w") as f:
        writer = csv.writer(f, lineterminator='\n')

        if isinstance(export_list[0], list): #多次元の場合
            writer.writerows(export_list)

        else:
            writer.writerow(export_list)

if __name__ == '__main__':
    main()
