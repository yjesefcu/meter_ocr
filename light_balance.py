import cv2
import numpy as np

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)
    h, w = img.shape[:2]
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))
    m = np.reshape(gray, [1, w * h])
    mean = m.sum() / (w * h)
    print 'mean: {}'.format(mean)
    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)


    return dst

if __name__ == '__main__':
    file = './test0309/1.jpg'
    blockSize = 24
    img = cv2.imread(file)
    cv2.imshow('origin', img)
    dst = unevenLightCompensate(img, blockSize)

    # result = np.concatenate([img, dst], axis=1)

    cv2.imshow('result', dst)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=3)
    # cv2.imshow('morphologyEx', dst)
    v = np.mean(dst)
    sigma=0.33
    lower = (int(max(0, (1.0 - sigma) * v)))
    upper = (int(min(255, (1.0 + sigma) * v)))
    edged = cv2.Canny(dst, lower, upper)
    cv2.imshow("edged", edged)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key=lambda x: x[1])
    yf = 100
    vec = []
    contador2 = 1
    # EDGE RECOGNITION-------------------------------------------------------------------------------------------------------
    consumo = 0
    edged_copy = img.copy()
    sp = edged_copy.shape
    height = sp[0]  # height(rows) of image
    width = sp[1]
    xfx = 1
    for (c, _) in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 11 and h > 14 and w < 100:
            cv2.rectangle(edged_copy, (x, y), (x + w, y + h), (255, 0, 0), 1, 4)
            if (x - xfx) > 10:
                if contador2 < 6:
                    xfx = x + w
                    yf = y
                    roi = dst[y:y + h, x:x + w]
                    guardar = roi.copy()
                    # cv2.imshow("roi",roi)
                    contador2 += 1
    cv2.imshow("edged_copy", edged_copy)
    cv2.waitKey(0)