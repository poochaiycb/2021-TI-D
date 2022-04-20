import cv2
import numpy as np
import time

# 计算平均值
# def average(*args):
#     return sum(args, 0.0) / len(args)

# L 计算公式
def L_data(ave_data, g):
    return (pow(ave_data / 14, 2) * g) / 4 * 3.14 * 3.14

def three_frame_differencing(videopath):
    cap = cv2.VideoCapture(videopath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    one_frame = np.zeros((height, width), dtype=np.uint8)
    two_frame = np.zeros((height, width), dtype=np.uint8)
    three_frame = np.zeros((height, width), dtype=np.uint8)
    '''参数设置'''
    i = 0
    flag = []  # 手动上一帧初始值为空
    ave_list = []  # 存储最后一次产生的所有数据
    cnt = 0   # 计数初始值
    start_time = 0
    g = 9.6942 # 小g参数  角度大L偏小g需要偏大，角度小g为10.4左右
    n = 34  # 检测次数

    '''循环'''
    while cap.isOpened():
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        one_frame, two_frame, three_frame = two_frame, three_frame, frame_gray
        abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
        _, thresh1 = cv2.threshold(abs1, 19.5, 255, cv2.THRESH_BINARY)  # 二值，大于 的为255，小于0

        abs2 = cv2.absdiff(two_frame, three_frame)
        _, thresh2 = cv2.threshold(abs2, 19.5, 255, cv2.THRESH_BINARY)

        binary = cv2.bitwise_and(thresh1, thresh2)  # 与运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erode = cv2.erode(binary, kernel)  # 腐蚀
        dilate = cv2.dilate(erode, kernel)  # 膨胀
        dilate = cv2.dilate(dilate, kernel)  # 膨胀

        contours, hei = cv2.findContours(dilate.copy(), mode=cv2.RETR_EXTERNAL,
                                         method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

        for contour in contours:
            if 100 < cv2.contourArea(contour) < 40000:
                x, y, w, h = cv2.boundingRect(contour)  # 找方框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
                # print(x, y, x + w, y + h)

        # cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        # cv2.imshow("binary", binary)
        # cv2.imshow("dilate", dilate)
        cv2.imshow("frame", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

        '''抛掉前五次开始记次数及测时间'''
        # AttributeError  IndexError
        if contours != []:  # 如果当前帧有框，令上一帧没有框
            flag = []


        elif contours == []:   # 如果当前帧没框并且上一帧也没有框则开始从第五次开始计数
            if flag ==[]:
                i += 1
                print('i_int = {}'.format(i))
                flag = [1]

                cnt += 1

            if i == 6:
                start_time = time.time()
                print(start_time)

        if cnt == n:  # 如果计数次数达到要求则停止计数并计算出所耗时间，并将数全部存进列表以便后面求平均值
            if flag == [1]:
                end_time = time.time()
                print(end_time)
                time_data = end_time - start_time
                print(time_data)
                ave_list.append(time_data)
                # print(ave_list)

        '''如果第n + 1次，将列表数值的平均值算出来代入算法公式'''
        if cnt == n + 1:
            ave_data = ave_list[0]
            # print(ave_data)
            L = L_data(ave_data, g)
            print('L = {}'.format(L))
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    three_frame_differencing(1)   # 指定摄像头0 or 1
