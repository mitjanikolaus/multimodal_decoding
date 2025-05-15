import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    image = cv2.imread('/home/mitja/Downloads/bird.jpg')
    plt.imshow(image)
    plt.show()
    def create_gaborfilter():
        filters = []
        num_filters = 2
        ksize = 30
        sigma = 3.0
        lambd = 10.0
        gamma = 0.5
        psi = 0
        for theta in np.arange(0, np.pi, np.pi / num_filters):
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
            kern /= 1.0 * kern.sum()
            filters.append(kern)
        return filters


    def apply_filter(img, filters):
        newimg = np.zeros_like(img)
        depth = -1
        for kern in filters:
            img_filter = cv2.filter2D(img, depth, kern)
            np.maximum(newimg, img_filter, newimg)

        return newimg

    gfilters = create_gaborfilter()
    image_g = apply_filter(image, gfilters)
    path = '/home/mitja/Downloads/bird_filtered.jpg'
    cv2.imwrite(path, image_g)
    img_gray_mode = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    path = '/home/mitja/Downloads/bird_grey.jpg'
    cv2.imwrite(path, img_gray_mode)
    # img_gray = cv2.cvtColor(image_g, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray_mode)
    plt.show()
