import cv2
import numpy as np
from PIL import ImageColor

base_colours = ['#EC2504', '#37AB65', '#00FF00', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7',
                '#FF6666']
frame_window = 'Frame-Window'
seeds_map_window = 'Seeds-Map-Window'
watershed_result_window = 'Watershed-Result-Window'


def waterShedFinal(img):
    markers = cv2.watershed(img, np.int32(seeds))

    img[markers == -1] = [0, 0, 255]
    for n in range(1, 10):
        img[markers == n] = ImageColor.getcolor(base_colours[n], "RGB")

    cv2.imshow(watershed_result_window, img)

    cv2.waitKey()


def click_event(event, x, y, _flags, _params):
    if event == cv2.EVENT_LBUTTONDOWN:
        val = int(chr(selected_key))
        points.append(((x, y), val))
        cv2.circle(seeds, (x, y), 7, (val, val, val), thickness=-1)


def main():
    global points
    global seeds
    global frame
    global selected_key
    selected_key = 49  # 1 en ASCII
    points = []
    # seeds = np.zeros((1198, 1198), np.uint8)
    seeds = np.zeros((720, 1280), np.uint8)
    cv2.namedWindow(frame_window)
    cv2.namedWindow(seeds_map_window)

    cap = cv2.VideoCapture(0)
    cv2.setMouseCallback(frame_window, click_event)

    while True:
        _, frame = cap.read()
        # frame = cv2.imread('./1.jpeg')
        frame_copy = frame.copy()
        seeds_copy = seeds.copy()

        for point in points:
            color = ImageColor.getcolor(base_colours[point[1]], "RGB")
            val = point[1] * 20

            x = point[0][0]
            y = point[0][1]
            cv2.circle(frame_copy, (x, y), 7, val, thickness=-1)
            cv2.circle(seeds_copy, (x, y), 7, val, thickness=-1)
            cv2.putText(frame_copy, str(point[1]), (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 3)

        cv2.imshow(frame_window, frame_copy)
        map = cv2.applyColorMap(seeds_copy, cv2.COLORMAP_COOL)
        cv2.imshow(seeds_map_window, map)

        key = cv2.waitKey(100) & 0xFF
        if key == 32:
            waterShedFinal(frame.copy())
            points = []
            # seeds = np.zeros((1198, 1198), np.uint8)
            seeds = np.zeros((720, 1280), np.uint8)

        if ord('1') <= key <= ord('9'):
            selected_key = key

        if key == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    main()
