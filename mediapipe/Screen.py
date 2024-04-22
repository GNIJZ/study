import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api

def grab_screen():
    # 获取整个屏幕的宽度和高度
    width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

    # 获取桌面窗口的设备上下文
    hwindc = win32gui.GetWindowDC(0)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()

    # 创建一个与设备上下文兼容的位图对象
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)

    # 将屏幕图像复制到位图对象中
    memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

    # 获取位图对象的像素数据
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    # 释放资源
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(0, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    # 将图像从BGRA颜色空间转换为RGB颜色空间，并返回
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

# # 截取整个屏幕
# grabbed_screen = grab_screen()
# cv2.imshow('Screen', grabbed_screen)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
