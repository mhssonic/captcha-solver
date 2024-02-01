import cv2
import numpy as np
import os
from keras import models


def resizeToNormal(image, x_mean_letter, y_mean_letter):
    y_letter, x_letter = image.shape
    if abs(y_mean_letter - y_letter) > 0.5 * y_mean_letter or abs(x_mean_letter - x_letter) > 0.5 * x_mean_letter:
        return None

    # Case when both dimensions are smaller
    if y_letter < y_mean_letter and x_letter < x_mean_letter:
        # Calculate border sizes
        top_border = (y_mean_letter - y_letter) // 2
        bottom_border = y_mean_letter - y_letter - top_border
        left_border = (x_mean_letter - x_letter) // 2
        right_border = x_mean_letter - x_letter - left_border

        # Add border to the image
        image = cv2.copyMakeBorder(image, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT,
                                   value=[0, 0, 0])

    # Case when both dimensions are larger
    elif y_letter >= y_mean_letter and x_letter >= x_mean_letter:
        # Calculate margins to crop
        top_margin = (y_letter - y_mean_letter) // 2
        left_margin = (x_letter - x_mean_letter) // 2

        # Crop the image
        image = image[top_margin:top_margin + y_mean_letter, left_margin:left_margin + x_mean_letter]

    # Case when height is larger but width is smaller
    elif y_letter >= y_mean_letter and x_letter < x_mean_letter:
        # Crop height
        top_margin = (y_letter - y_mean_letter) // 2
        image = image[top_margin:top_margin + y_mean_letter, :]

        # Add border to width
        left_border = (x_mean_letter - x_letter) // 2
        right_border = x_mean_letter - x_letter - left_border
        image = cv2.copyMakeBorder(image, 0, 0, left_border, right_border, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Case when height is smaller but width is larger
    elif y_letter < y_mean_letter and x_letter >= x_mean_letter:
        # Crop width
        left_margin = (x_letter - x_mean_letter) // 2
        image = image[:, left_margin:left_margin + x_mean_letter]

        # Add border to height
        top_border = (y_mean_letter - y_letter) // 2
        bottom_border = y_mean_letter - y_letter - top_border
        image = cv2.copyMakeBorder(image, top_border, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return image

def invert_colors_if_white(image):
    # Calculate the percentage of white pixels
    white_pixels = np.sum(image > 128)
    total_pixels = image.shape[0] * image.shape[1]
    white_percentage = (white_pixels / total_pixels) * 100

    # If the majority of the image is white, invert the colors
    if white_percentage > 30:
        image = 255 - image
        # imshow(image)
    return image


def imshow(img):
    cv2.imshow('image', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()



def handling_image(path, letter_count):
    img = cv2.imread(path)
    if img is None:
        return None
    y_img, x_img, _ = img.shape
    x_mean_letter = int(x_img / letter_count)
    y_mean_letter = y_img

    image = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY +
                                 cv2.THRESH_OTSU)

    thresh1 = cv2.bitwise_not(thresh1)

    kernel = np.ones((2, 2), np.uint8)  # Adjust the kernel size as needed

    # Apply erosion to shrink the black regions
    shrunken_image = cv2.erode(thresh1, kernel, iterations=1)
    # imshow(shrunken_image)
    shrunken_image = invert_colors_if_white(shrunken_image)
    height, width = shrunken_image.shape
    x_shrunk, y_shrunk = 2, 4
    cut_image = shrunken_image[y_shrunk:height - y_shrunk, x_shrunk: width - x_shrunk]
    bordered_image = cv2.copyMakeBorder(cut_image, y_shrunk, y_shrunk, x_shrunk, x_shrunk,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # bordered_image = cv2.rectangle(shrunken_image, (0, 0), (width - 1, height - 1), 0, 6)

    # reader = easyocr.Reader(['en'])

    contours, _ = cv2.findContours(bordered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_array = []
    x_image_array = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        # print(cx, cy)

        x, y, w, h = cv2.boundingRect(contour)
        letter_image = shrunken_image[y:y + h, x:x + w]
        # You can save the extracted letter images or process them further as needed
        # cv2.imwrite('letter_{}.png'.format(len(letters)), letter_image)
        # text = pytesseract.image_to_string(letter_image)
        # print("letters is :" + text)

        # result = ocrolib.read_text(letter_image)

        border = resizeToNormal(letter_image, x_mean_letter, y_mean_letter)
        if border is None:
            continue

        # result = reader.readtext(border)
        # print(result)
        image_array.append(border)
        x_image_array.append(cx)
        # imshow(border)

    x_image_array = np.array(x_image_array)
    sort = np.argsort(x_image_array)
    try:
        image_array = np.array(image_array)
    except:
        print([x.shape for x in image_array])
        return None
    if len(image_array) != letter_count:
        # imshow(bordered_image)
        return None
    return image_array[sort]


def save_images(base_path, letter_count):
    for name in os.listdir(base_path):
        path = base_path + '/' + name
        image_array = handling_image(path, letter_count)
        if image_array is not None:
            for img in image_array:
                cv2.imwrite('letter_{}.jpg'.format(len(image_array)), img)
                imshow(img)
        else:
            print("lost image: " + path)


def rename_images(base_path, dif_image, same_img):
    for i in range(dif_image):
        path = base_path + '/' + str(i) + '_0.jpg'
        img = cv2.imread(path)
        imshow(img)
        string = input()
        for j in range(same_img):
            file = base_path + '/' + str(i) + '_' + str(j) + '.jpg'
            os.rename(file, base_path + '/' + string + '_' + str(j) + '.jpg')


def img_to_letter(images):
    model = models.load_model('my_model.keras')
    # imshow(images[3])
    predicts = model.predict(images)
    p = np.array(predicts)
    text = ""
    for i in p:
        number = i.argmax()
        text += chr(ord('A') + number)
    return text


def image_to_text(path: str, letter_count: int):
    images = handling_image(path, letter_count)
    if images is None:
        return ""
    return img_to_letter(images / 255)


if __name__ == '__main__':
    images = os.listdir('data')
    images = ["data/" + image for image in images]
    right = 0
    wrong = 0
    filter_problem = 0
    for image in images:
        print(image + ":")
        text = image_to_text(image, 6)
        print(text)
        print("------------")
        if text == "":
            filter_problem += 1
        elif text == image[5:11]:
            right += 1
        else:
            wrong += 1
    print(f"wrong= {wrong}, right={right}, filter_problem={filter_problem}")