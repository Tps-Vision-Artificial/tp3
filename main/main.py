import cv2 as cv
import numpy as np
from grabCut import grabcut
from waterShed import watershed

def mainGrabCut():
    # iniciamos la capturadora con el nombre cap
    cap = cv.VideoCapture(0)
    window_name = 'Window'
    threshold_trackbar_name = 'Treshold Trackbar'
    radius_trackbar_name = 'Radius'
    slider_max = 151
    number = 0
    cv.namedWindow(window_name)
    # cap = cv.VideoCapture()
    biggest_contour = None

    create_trackbar(threshold_trackbar_name, window_name, slider_max)
    create_trackbar(radius_trackbar_name, window_name, 30)

    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)

        # 1
        gray_frame = apply_color_convertion(frame=frame, color=cv.COLOR_BGR2GRAY)

        # 2
        trackbar_val = get_trackbar_value(trackbar_name=threshold_trackbar_name, window_name=window_name)

        _, threshold_frame = threshold(frame=gray_frame, slider_max=slider_max, trackbar_value=trackbar_val)
        # 3

        radius = get_trackbar_value(trackbar_name=radius_trackbar_name, window_name=window_name)
        frame_denoised = denoise(frame=threshold_frame, method=cv.MORPH_ELLIPSE, radius=radius)

        # 4 Contours
        contours = get_contours(frame=frame_denoised, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        final_frame = apply_color_convertion(frame=frame_denoised, color=cv.COLOR_GRAY2BGR)
        final_frame_color = frame

        # cv.imshow('Frame', frame)
        cv.imshow('Gray', gray_frame)
        cv.imshow('Threshold', threshold_frame)
        cv.imshow('Denoised', frame_denoised)
        cv.imshow('Window', final_frame_color)

        if cv.waitKey(1) % 256 == 32:
            img_name = "GC_picture_{}.png".format(number)
            cv.imwrite(img_name, final_frame_color)
            number += 1
            grabcut(final_frame_color)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def mainWaterShed():
    # iniciamos la capturadora con el nombre cap
    cap = cv.VideoCapture(0)
    window_name = 'Window'
    threshold_trackbar_name = 'Treshold Trackbar'
    radius_trackbar_name = 'Radius'
    slider_max = 151
    number = 0
    cv.namedWindow(window_name)
    # cap = cv.VideoCapture()
    biggest_contour = None

    create_trackbar(threshold_trackbar_name, window_name, slider_max)
    create_trackbar(radius_trackbar_name, window_name, 30)

    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)

        # 1
        gray_frame = apply_color_convertion(frame=frame, color=cv.COLOR_BGR2GRAY)

        # 2
        trackbar_val = get_trackbar_value(trackbar_name=threshold_trackbar_name, window_name=window_name)

        _, threshold_frame = threshold(frame=gray_frame, slider_max=slider_max, trackbar_value=trackbar_val)
        # 3

        radius = get_trackbar_value(trackbar_name=radius_trackbar_name, window_name=window_name)
        frame_denoised = denoise(frame=threshold_frame, method=cv.MORPH_ELLIPSE, radius=radius)

        # 4 Contours
        contours = get_contours(frame=frame_denoised, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        final_frame = apply_color_convertion(frame=frame_denoised, color=cv.COLOR_GRAY2BGR)
        final_frame_color = frame

        # cv.imshow('Frame', frame)
        cv.imshow('Gray', gray_frame)
        cv.imshow('Threshold', threshold_frame)
        cv.imshow('Denoised', frame_denoised)
        cv.imshow('Window', final_frame_color)

        if cv.waitKey(1) % 256 == 32:
            img_name = "WS_picture_{}.png".format(number)
            cv.imwrite(img_name, final_frame_color)
            number += 1
            watershed(final_frame_color)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def create_trackbar(trackbar_name, window_name, slider_max):
    cv.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar)


def on_trackbar(val):
    pass


def get_trackbar_value(trackbar_name, window_name):
    return int(cv.getTrackbarPos(trackbar_name, window_name) / 2) * 2 + 3


def denoise(frame, method, radius):
    kernel = cv.getStructuringElement(method, (radius, radius))
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    return closing

def apply_color_convertion(frame, color):
    return cv.cvtColor(frame, color)


def threshold(frame, slider_max, trackbar_value):
    # return cv.adaptiveThreshold(frame, slider_max, adaptative, binary, trackbar_value, 0)
    return cv.threshold(frame, trackbar_value, slider_max, cv.THRESH_BINARY_INV)


def get_contours(frame, mode, method):
    contours, hierarchy = cv.findContours(frame, mode, method)
    return contours


def get_biggest_contour(contours):
    max_cnt = contours[0]
    for cnt in contours:
        if cv.contourArea(cnt) > cv.contourArea(max_cnt):
            max_cnt = cnt
    return max_cnt

def grabcut(img):
    mask = np.zeros(img.shape[:2], np.uint8)

    # These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # usamos roi para agarrar el rect
    rect = cv.selectROI("img", img, fromCenter=False, showCrosshair=True)
    print(mask)

    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

    print(mask)
    # ????????????
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    print(mask2)

    # ?????????????
    img = img * mask2[:, :, np.newaxis]

    cv.imshow("img", img)
    cv.waitKey()


mainWaterShed()
