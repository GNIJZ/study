import keyboard
import win32api
from pynput.mouse import Controller
from Screen import grab_screen
import cv2
import mediapipe as mp
import time


mouse = Controller()
# 将鼠标移动到指定位置（x, y）
# mouse.position = (2190, 1230)
nose_x=None
nose_y=None
def draw_nose(image, x, y):
    # 定义圆圈的颜色和半径
    color = (0, 255, 0)  # 绿色
    radius = 5  # 圆圈半径

    # 在图像上绘制一个圆圈来表示鼻子位置
    cv2.circle(image, (x, y), radius, color, -1)  # -1 表示实心圆

mpPose=mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   model_complexity=1,
                   smooth_landmarks=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5
                   )
# 获取屏幕的宽度和高度
screen_width = win32api.GetSystemMetrics(0)
screen_height = win32api.GetSystemMetrics(1)

print(screen_width,screen_height)
while True:
    if keyboard.is_pressed('q'):
        break
    img = grab_screen()
    # img = grab_screen()
    results = pose.process(img)

    if results.pose_landmarks is not None:
        nose_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE]

     # 计算检测框的实际大小
        print(nose_landmark.x,nose_landmark.y )
        nose_x = int(nose_landmark.x * screen_width)

        nose_y = int(nose_landmark.y * screen_height)

        #print(nose_x, nose_y)
        mouse.position = (nose_x, nose_y)

    else:
        # 处理未检测到姿势关键点的情况
        #print("No pose landmarks detected")
        nose_x=None
        nose_y=None
    # 在这里可以使用 nose_x 和 nose_y 进行后续处理

    # 等待一段极短的时间来检查键盘输入
    time.sleep(0.001)




