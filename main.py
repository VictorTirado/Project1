import numpy as np
import cv2

def matchingimage(input_img, target, threshold):
    inputrows, inputcols = input_img.shape
    targetrows, targetcols = target.shape

    img_founded = False

    # Crear matching image
    matchingrows = inputrows - targetrows + 1
    matchingcols = inputcols - targetcols + 1
    matching = np.zeros((matchingrows, matchingcols), dtype=np.float32)

    for i in range(0, matchingrows):
        for j in range(0, matchingcols):
            pixelSum = 0
            for i2 in range(0, targetrows):
                for j2 in range(0, targetcols):
                    pixelMatching = (int(target[i2, j2]) - int(input_img[i+i2, j+j2])) ** 2
                    pixelSum += pixelMatching
            matching[i,j] = pixelSum

    # Normalizar
    matching /= matching.max()
    matching *= 255

    minVals = []
    # Comprobar si el target esta en la imagen
    if matching.min()/matching.max() < threshold:
        img_founded = True
        for i in range(0, matchingrows):
            for j in range(0, matchingcols):
                if matching[i,j] == matching.min():
                    minVals.append([i,j])

    return img_founded, matching, minVals

def getResult(founded):
    if founded == True:
        imgFound = np.zeros((40,245,3),np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imgFound,"TARGET FOUND",(5,30),font,1,(0,255,0),2)
        return imgFound
    else:
        imgNotFound = np.zeros((40,320,3),np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imgNotFound, "TARGET NOT FOUND", (5, 30), font, 1, (0, 0, 255), 2)
        return imgNotFound

def drawQuads(img, minVals, targetshape, scale):
    for i in minVals:
        cv2.rectangle(img, ((int(i[1]/scale)) -1, int((i[0])/scale) -1), (int(i[1]/scale) + targetshape[1], int(i[0]/scale) + targetshape[0]), (0,255,0))

def scaleImages(img, target, scale):
    if scale != 1.0:
        img = cv2.resize(img,(0,0),fx=scale,fy=scale);
        target = cv2.resize(target,(0,0),fx=scale,fy=scale);

    return img, target

if __name__ == '__main__':

    image = input("Input image: ")
    template = input("Target image: ")
    threshold = float(input("Detection threshold: "))
    img_color = cv2.imread(image, cv2.IMREAD_ANYCOLOR)
    img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    target_color = cv2.imread(template,cv2.IMREAD_ANYCOLOR)
    target = cv2.imread(template,cv2.IMREAD_GRAYSCALE)

    #scale = 0.0
    #while scale == 0.0:
    #    scale = float(input("Image Scale(must be > 0.0): "))

    scale = 1.0
    img, target = scaleImages(img, target, scale)

    isFounded, matching_img, minVals = matchingimage(img, target, threshold)
    resultimg = getResult(isFounded)

    if isFounded == True:
        drawQuads(img_color, minVals, target_color.shape, scale)


    cv2.imshow("Input Image",img_color)
    cv2.imshow("Target",target_color)
    cv2.imshow("Matching Map", np.uint8(matching_img))
    cv2.imshow("Result", resultimg)

    cv2.waitKey(0)