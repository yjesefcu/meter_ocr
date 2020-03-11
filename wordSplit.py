# -*- coding:utf-8 -*-
import cv2
import numpy as np
import utils
import preprocess


def column_shadow(img, color=255):
    # 垂直投影，只返回投影后的白色(color=255)或黑色(color=0)点位个数
    height, width = img.shape[:2]
    v = [0] * width
    a = 0

    # 垂直投影：统计并存储每一列的黑点数
    for x in range(0, width):
        for y in range(0, height):
            if img[y, x] == color:
                a = a + 1
            else:
                continue
        v[x] = a
        a = 0
    return v


def line_shadow(img, color=255):
    # 水平投影，只返回投影后的白色(color=255)或黑色(color=0)点位个数
    height, width = img.shape[:2]
    v = [0] * width
    a = 0

    # 垂直投影：统计并存储每一列的黑点数
    for x in range(0, width):
        for y in range(0, height):
            if img[y, x] == color:
                a = a + 1
            else:
                continue
        v[x] = a
        a = 0
    return v


def _after_column_split(v, width, height):

    def _cut_multi_words(v, pos): # 将多个字符切割
        x0, x1 = pos
        d = 20
        groups = []
        widths = []
        while len(groups) <= 1:
            startIndex = -1
            groups = []
            widths = []
            for i in range(x0, x1+1):
                v[i] -= d
                if v[i] <= 0:
                    v[i] = 0
                    if startIndex >= 0:
                        groups.append((startIndex, i-1))
                        widths.append(i-startIndex)
                        startIndex = -1
                elif startIndex < 0:
                    startIndex = i
            if len(groups) > 1:
                break
            if d > 1:
                d -= 1
        return groups, widths

    wordPos = []  # 记录每个字符的起止点位 [(start,end),(start,end)]
    wordWidth = []  # 每个字符所占宽度
    maxCount = max(v)  #
    startIndex = -1
    # 通过黑色占比去掉一些燥点
    _show_column_split_img(v, width, height, '{}'.format(width))
    for i in range(0, len(v)):
        if (v[i] / float(maxCount)) < (1 / float(20)):  # 将黑色点数占比不到1/7的去除，避免一些燥点被算进去
            v[i] = 0
            if startIndex >= 0:
                if i-startIndex > (height * 1.2):
                    tmpGroups, tmpWidths = _cut_multi_words(v, (startIndex, i-1))
                    wordPos.extend(tmpGroups)
                    wordWidth.extend(tmpWidths)
                else:
                    wordPos.append((startIndex, i - 1))
                    wordWidth.append(i - startIndex)
                startIndex = -1
        else:
            if startIndex < 0:
                startIndex = i
    if startIndex >= 0:  # 如果最后一列仍是字符
        wordPos.append((startIndex, i - 1))
        wordWidth.append(i - startIndex)
    if len(wordWidth) == 0:
        return []
    # 根据wordPos每个字符的宽度，判断是否有效字符
    actualWordsPos = []  # 最终实际的字符的x坐标
    medianWidth = np.median(wordWidth)  # 宽度的中位数
    for pos in wordPos:
        w = pos[1] - pos[0]
        if w > medianWidth / 2.5:  # 如果宽度小于宽度中位数的一半，则不认为是字符
            # 如果宽度大于高度的1.5倍，则可能是两个字符组成，需要再次进行分割
            actualWordsPos.append(pos)
    return actualWordsPos


def _show_column_split_img(v, width, height, name='chuizhi'):
    # 创建空白图片，绘制垂直投影图
    emptyImage = np.full((height, width, 3), 0, np.uint8)
    for x in range(0, width):
        for y in range(0, v[x]):
            b = (255, 255, 255)
            emptyImage[y, x] = b
    cv2.imshow(name, emptyImage)


def _show_line_split_img(v, width, height, name='shuiping'):
    # 绘制水平投影图
    emptyImage1 = np.full((height, width, 3), 0, dtype=np.uint8)
    for y in range(0, height):
        for x in range(0, v[y]):
            b = (255, 255, 255)
            emptyImage1[y, x] = b
    cv2.imshow(name, emptyImage1)


def column_split(img, show=False, min_word_count=5):
    height, width = img.shape[:2]

    def _cut_multi_words(v, pos): # 将多个字符切割
        x0, x1 = pos
        d = 10
        groups = []
        widths = []
        while len(groups) <= 1:
            startIndex = -1
            groups = []
            widths = []
            for i in range(x0, x1+1):
                v[i] -= d
                if v[i] <= 0:
                    v[i] = 0
                    if startIndex >= 0:
                        groups.append((startIndex, i-1))
                        widths.append(i-startIndex)
                        startIndex = -1
                elif startIndex < 0:
                    startIndex = i
            if len(groups) > 1:
                break
            if d > 1:
                d -= 1
        return groups, widths

    # 垂直分割
    v = column_shadow(img, color=255)
    _show_column_split_img(v, width, height, 'chuizhi_0')
    actualWordsPos = _after_column_split(v, width, height)
    index = 1
    while len(actualWordsPos) < min_word_count:
        for i in range(0, len(v)):
            v[i] -= 10
            if v[i] < 0:
                v[i] = 0
        if max(v) == 0:
            break
        actualWordsPos = _after_column_split(v, width, height)
        _show_column_split_img(v, width, height, 'chuizhi_{}'.format(index))
        index += 1

    if len(actualWordsPos) == 0:
        return []
    if show:
        # 创建空白图片，绘制垂直投影图
        emptyImage = np.full((height, width, 3), 0, np.uint8)
        for x in range(0, width):
            for y in range(0, v[x]):
                b = (255, 255, 255)
                emptyImage[y, x] = b
        for w in actualWordsPos:
            cv2.rectangle(emptyImage, (w[0], 0), (w[1], height), (0, 0, 255), 2)  # 用矩形显示最终字符
        cv2.imshow('chuizhi', emptyImage)
    # 将字符取出
    return actualWordsPos


def line_split(word, whole_split=False, show_window=None):
    # 对单个字符进行行分割, whole_split:表示是整张图片的分割，还是单个字符的分割
    # 水平投影  #统计每一行的黑点数
    height, width = word.shape[:2]
    v = [0] * height
    a = 0
    # 统计水平方向的白色个数
    for y in range(0, height):
        for x in range(0, width):
            if word[y, x] == 255:
                a = a + 1
            else:
                continue
        v[y] = a
        a = 0
    # 将黑色点太少的去除
    wordPos = []  # 记录每个字符的起止点位 [(start,end),(start,end)]
    wordWidth = []  # 每个字符所占宽度
    maxCount = max(v)  #
    if maxCount == 0:
        return None
    startIndex = -1
    # 通过黑色占比去掉一些燥点
    for i in range(0, len(v)):
        if (whole_split and v[i] < (width / 20)) or \
                (v[i] / float(maxCount)) < (1 / float(20)):  # 将黑色点数占比不到1/10的去除，避免一些燥点被算进去
            v[i] = 0
            if startIndex >= 0:
                wordPos.append((startIndex, i - 1))
                wordWidth.append(i - startIndex)
                startIndex = -1
        else:
            if startIndex < 0:
                startIndex = i
    if startIndex >= 0:  # 如果最后一列仍是字符
        wordPos.append((startIndex, i - 1))
        wordWidth.append(i - startIndex)
    maxIndex = np.argmax(wordWidth) # 宽度最大的占比，作为最终的字符位置
    if show_window:
        # 绘制水平投影图
        emptyImage1 = np.full((height, width, 3), 0, dtype=np.uint8)
        for y in range(0, height):
            for x in range(0, v[y]):
                b = (255, 255, 255)
                emptyImage1[y, x] = b
        cv2.imshow(show_window, emptyImage1)
    return wordPos[maxIndex]


def word_column_split(word): # 对单个字符在进行一次垂直分割，取出不必要的左右的空间
    v, groups, widths = preprocess.column_shadow(word, color=255)
    if len(widths) == 0:
        return None
    # 取占比最宽的groups
    maxIndex = np.argmax(widths) # 宽度最大的占比，作为最终的字符位置
    return groups[maxIndex]

def pre_process(img, show=False):
    # img = preprocess.convert_red_to_black(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    gris = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img = utils.custom_threshold(gris)
    if show:
        cv2.imshow("thresh", img)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 形态学处理:定义矩形结构
    # img = cv2.erode(img, kernel, iterations=2)  # 腐蚀
    # if show:
    #     cv2.imshow("erode", img)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 形态学处理:定义矩形结构
    # img = cv2.dilate(img, kernel, iterations=2)
    # if show:
    #     cv2.imshow("dilate", img)
    return img


# def after_process(words):
    # 对选取处理的字符进行后处理
    # 后处理的逻辑，对间隔太近


def _rect_digital(img, show=False):
    origin_height, origin_width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (x, y, w, h) = preprocess.rect_boundary(gray)
    digital_rect = img[y:y + h, x:origin_width]  # 取出数字区域
    # 步骤2：判断红色区域
    red_pos = preprocess.get_meter_red_area(digital_rect)
    if red_pos:  # 如果存在数字区域，将数字区域作为结尾
        redy0, redy1 = red_pos
        w = redy1
    if show:
        digital_rect = digital_rect[0:h, 0:w]  # 取出数字区域
        cv2.imshow('digital rect', digital_rect)
    return (x, y, w, h)


def column_slit2(thresh, words=5, show=False, initial_ratio=3): # 垂直分割：
    def _group(l):
        groups = []
        start_index = -1
        group_height = []
        for i in range(0, width):
            if l[i] > 0 and start_index < 0:
                start_index = i
            elif (l[i] <= 0 and start_index >= 0) or (i == width - 1 and start_index >= 0):
                groups.append((start_index, i - 1))
                group_height.append(i - start_index)
                start_index = -1
        groups, heights = _filter(l, groups, group_height)
        return groups, heights

    def _filter(l, groups, widths): # 将过小的数据去除
        tmp_groups = []
        tmp_widths = []
        for i in range(0, len(groups)):
            g = groups[i]
            w = widths[i]
            if w > width/40 and max(l[g[0]:g[1]+1]) > height/10:
                tmp_groups.append(g)
                tmp_widths.append(w)
        return tmp_groups, tmp_widths

    v = column_shadow(thresh, color=255)
    height, width = thresh.shape[:2]
    tmp = []
    diff = int(height / float(initial_ratio))
    for i in v:
        tmp.append(i - diff)
    tmp_groups = []
    groups, widths = _group(tmp)
    flag = len(groups) >= words
    step = 10
    while diff > 0:
        tmp_groups = groups
        tmp = []
        for i in v:
            if i - diff < 0:
                tmp.append(0)
            else:
                tmp.append(i - diff)
        # if show:
        #     _show_column_split_img(tmp, width, height, 'tmp{}'.format(diff))
        #     cv2.waitKey(0)
        groups, widths = _group(tmp)
        if not flag:
            if len(groups) >= words:
                flag = True
        elif len(groups) < words:
            break
        diff -= step
        if step > 5:
            step -= 1
    if show:
        _show_column_split_img(tmp, width, height, 'tmp')

    # 创建空白图片，绘制垂直投影图
    emptyImage = np.full((height, width, 3), 0, np.uint8)
    if show:
        for x in range(0, width):
            for y in range(0, v[x]):
                b = (255, 255, 255)
                emptyImage[y, x] = b
        for w in tmp_groups:
            cv2.rectangle(emptyImage, (w[0], 0), (w[1] + 1, height + 1), (0, 0, 255), 2)  # 用矩形显示最终字符
        cv2.imshow('finaly', emptyImage)
        # cv2.waitKey(0)
    return tmp_groups


def _validate_by_width(thresh, rects, words=6): # 通过字符的宽度判断是否包含多个字符
    height, width = thresh.shape[:2]
    avgWidth = width/words # 将图片宽度除以字符个数
    toInserts = []
    for i in range(0, len(rects)):
        (x, y, w, h) = rects[i]
        count = int(w/float(avgWidth) + 0.7) #
        if count > 1: # 宽度>间隔的1.4被，说明包含两个字符以上
            tmpGroups = column_slit2(thresh[y:y+h, x:x+w], words=count, initial_ratio=1.2)
            toInserts.append((i, tmpGroups))
    toInserts.reverse() # 反转，从后面的数据开始插入
    for toInsert in toInserts:
        count = len(rects)
        index, tmpGroups = toInsert
        (x, y, w, h) = rects[index]
        for j in range(0, len(tmpGroups)):
            r = (tmpGroups[j][0]+x, y, (tmpGroups[j][1] - tmpGroups[j][0]), h)
            if j == 0: # 替换原来的
                rects[j+index] = r
            elif index == count - 1: # 在列表的最后，直接插入
                rects.append(r)
            else: # 在列表的中间，用insert插入
                rects.insert(index+1, r)
    return rects


def _validate_by_interval(rects): # 通过字符间隔判断字符有效性
    ws = [] # 所有矩形的宽度
    intervals = []
    for i in range(0, len(rects)):
        ws.append(rects[i][2])
        if i < len(rects) - 1:
            v = (rects[i + 1][0] + rects[i + 1][2] / 2) - (rects[i][0] + rects[i][2] / 2)
            intervals.append(v)
    avgw = np.average(ws)
    results = []
    i = 0
    while i < len(rects) - 1:
        # 用矩形的中间计算两个矩形之间的间隔
        if i == len(rects) - 1:
            break
        w = (rects[i+1][0] + rects[i+1][2]/2) - (rects[i][0] + rects[i][2]/2)
        if w > avgw * 1.5:
            results.append(rects[i])
            results.append(rects[i+1])
        else:
            # 间隔过近，判断哪个是正确的字符
            if rects[i][2] > rects[i+1][2]:
                results.append(rects[i])
            else:
                results.append(rects[i+1])
        i += 2
    if i == len(rects) - 1: # 还有最后一个字符
        prev = results[-1]
        w = (rects[i][0] + rects[i][2]/2) - (prev[0] + prev[2]/2)
        if w > avgw * 1.5:
            results.append(rects[i])

    return results


def tight_word(gray, rect):
    def _get_row(timg):
        v, groups, widths = preprocess.line_shadow(timg)
        index = np.argmax(widths)
        return groups[index]

    def _get_col(timg):
        v, groups, widths = preprocess.column_shadow(timg)
        index = np.argmax(widths)
        return groups[index]

    x, y, w, h = rect
    img = gray[y:y+h, x:x+w]
    threshold = utils.custom_threshold(img)
    # if w > h/1.8: # 先垂直再水平
    #     x0, x1 = _get_col(threshold)
    #     y0, y1 = _get_row(threshold[0:h, x0:x1])
    # else: # 先水平再垂直
    y0, y1 = _get_row(threshold)
    x0, x1 = _get_col(threshold[y0:y1+1, 0:w])
    y2, y3 = _get_row(threshold[y0:y1+1, x0:x1+1])
    cv2.rectangle(threshold, (x0, y0+y2), (x1+1, y0+y2+y3+1), (255, 255, 255), 2)  # 用矩形显示最终字符
    cv2.imshow('tight{}'.format(x), threshold)
    return (x0+x, y0+y2+y, x1-x0+1, y3-y2+1)


def img_to_words(img, show=False, words=6):
    # # 将图片分割成字符
    # 步骤1：识别出数字区域
    (x1, y1, w1, h1) = _rect_digital(img)
    digital1 = img[y1:y1+h1, x1:x1+w1]
    if show:
        cv2.imshow('first digital area', digital1)
    # 步骤1：取出数字区域
    (x2, y2, w2, h2) = _rect_digital(digital1)
    x = x1 + x2
    y = y1 + y2
    w = w2
    h = h2
    # 步骤2：图像预处理
    # closed = pre_process(img, show)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波
    closed = utils.custom_threshold(gray)
    closed = closed[y:y+h, x:x+w]
    # 步骤3：进行列分割，分割的图像基于数字区
    xWords = column_slit2(closed, words=words, show=show)
    # xWords = column_split(closed, show) # 垂直的字符位置
    cv2.imshow('closed', closed)
    if len(xWords) == 0:
        print 'column_split return empty'
        return []
    tmp = []  # 最终的字符的(x,y,w,h)
    # 步骤4：将列分割的结果进行行分割，分解出最终的数字
    wordHeights = []
    for i in range(0, len(xWords)):
        xPos = xWords[i]
        # 对每个字符进行水平分割
        wordRect = closed[0:h, xPos[0]:xPos[1]+1]
        # wordRect = utils.custom_threshold(wordRect)
        yPos = line_split(wordRect, show_window=None)
        if yPos is None:
            continue
        # 对字符再做一次垂直头像，取出左右不必要的空间
        r1 = wordRect[yPos[0]:yPos[1]+1, 0:wordRect.shape[1]]
        # r1 = utils.custom_threshold(r1)
        xPos2 = word_column_split(r1)
        if xPos2 is None:
            continue
        # x1 = xPos[0] + x
        # y1 = yPos[0] + y
        # w1 = xPos[1] - xPos[0]
        # h1 = yPos[1] - yPos[0]
        x1 = xPos[0] + xPos2[0] + x
        y1 = yPos[0] + y
        w1 = xPos2[1] - xPos2[0] + 1
        h1 = yPos[1] - yPos[0] + 1
        tmp.append((x1, y1, w1, h1))
        wordHeights.append(h1)
    # 步骤5：对切割结果过滤，去除不可能是数字的部分：取字符高度的中位数，去除高度小于高度中位数1/3的字符
    heightMedium = np.average(wordHeights)
    wordRects = []
    for t in tmp:
        (x1, y1, w1, h1) = t
        if h1 >= heightMedium / 2:
            wordRects.append(t)
    # 步骤6：通过字符的宽度判断是否包含多个字符
    wordRects = _validate_by_width(closed, wordRects, words=words)
    # 步骤7：通过间隔判断不合理的数据
    wordRects = _validate_by_interval(wordRects)
    # 步骤8：重新二值化后再次收缩字符范围
    for i in range(0, len(wordRects)):
        r = wordRects[i]
        r1 = tight_word(gray, r)
        wordRects[i] = r1
    if show:
        for (x1, y1, w1, h1) in wordRects:
            cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)  # 用矩形显示最终字符
        cv2.imshow('words', img)
    return wordRects # 返回每个字符的(x,y,w,h)


if __name__ == '__main__':
    # img = cv2.imread('./test0309/34.jpg') # 4,29,34,36,37,44,45,46
    img = cv2.imread('./test0311/7.jpg') # 34,37,45,46
    # img = cv2.imread('./area/13.png') # 5，18，31，42，51，54，55
    angle, img = utils.correct_skew(img, is_gray=False)
    cv2.imshow('skew', img)
    oriHeight, oriWidth = img.shape[:2]
    resizedHeight = int(oriHeight / (oriWidth / float(800)))
    img = cv2.resize(img, (800, resizedHeight)) # 将图片宽度固定为800
    img_to_words(img, show=True, words=6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()