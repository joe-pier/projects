import cv2
import matplotlib.pyplot as plt
import numpy as np

############################################################################
edge_detecting_x = np.array([[-0.25, 0, 0.25],
                             [-0.5, 0, 0.5],
                             [-0.25, 0, 0.25]
                             ])

edge_detecting_y = np.array([[0.25, 0.5, 0.25],
                             [0, 0, 0],
                             [-0.25, -0.5, -0.25]
                             ])
############################################################################

imagine_ = 'immagini/salvador.jpg'

############################################################################
crop = 100

image = cv2.imread(imagine_)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (crop, crop))
resized_ = cv2.imwrite('outputs/resized.jpg', image)

result_0 = cv2.filter2D(image, -1, edge_detecting_x)
cv2.imwrite('outputs/temp_X.jpg', result_0)

result_1 = cv2.filter2D(image, -1, edge_detecting_y)
cv2.imwrite('outputs/temp_Y.jpg', result_1)

############################################################################

power_result_0 = np.power(result_0, 2)
power_result_1 = np.power(result_1, 2)
somma = np.add(power_result_1, power_result_0)
radice = np.power(somma, 0.7)

############################################################################

cv2.imwrite('outputs/output.jpg', radice)


############################################################################

def LeastEdgy(E):
    least_E = np.zeros(np.shape(E))
    least_E[1] = E[1]
    dirs = np.zeros(np.shape(E))
    m, n = np.shape(E)

    for i in reversed(range(-1, m - 1)):
        for j in range(0, n - 1):
            j1, j2 = max(0, j - 1), min(j + 1, n - 1)
            j3, j4= max(0, j - 2), min(j + 2, n - 1)
            b = least_E[i + 1][j1]
            a = least_E[i + 1][j]
            c = least_E[i + 1][j2]
            u = least_E[i + 1][j3]
            t = least_E[i + 1][j4]
            h = [u, b, a, c, t]

            f = np.array(h)
            e = np.min(f)
            dir_ = np.argmin(f)

            least_E[i][j] += e *1
            least_E[i][j] += E[i][j] *0.8
            dirs[i][j] = -2 if dir_ == 0 else -1 if dir_ == 1 else 0 if dir_ == 2 else 1 if dir_==3 else 2

    cv2.imwrite('outputs/LOWEST_ENERGY_PATH.jpg', least_E)

    return [dirs, least_E]


############################################################################

direzioni = LeastEdgy(radice)
dir_ = direzioni[0]
immagine = direzioni[1]


############################################################################

def get_path(dirs, j):
    m = np.shape(dirs[0])
    js = np.zeros(m)

    js[0] = j
    temp = []

    for i in range(1, m[0]):
        h = js[i - 1]
        js[i] = max(0, js[i - 1] + dirs[i - 2, int(h)])
        temp.append(i)

    miao = list(zip(temp, js))
    return miao


############################################################################

# rileggo l'immagine iniziale per averla colorata  e croppata
image_fin = cv2.imread(imagine_)
image_resized_color = cv2.resize(image_fin, (crop, crop))
cv2.imwrite('outputs/resized_colores.jpg', image_resized_color)

'''
############################################################################
def mark(img, path):
    img = cv2.imread(img)

    #range(0,len(path))
    for k in range(30,31): # RANGE IMPORTANTE

        for i, j in path[k]:
            img[0, k] = (255, 0, 255)
            img[i, int(j)] = (255, 0, 255)

    cv2.imwrite('outputs/resized_colores.jpg', img)


paths = []
for i in range(0,crop):
    path_= get_path(dir_,i)
    paths.append(path_)

prova = mark('outputs/resized_colores.jpg', paths)

############################################################################
'''


############################################################################
def mark(img, path):
    img = cv2.imread(img)

    for i, j in path:

        img[i - 1, int(j)] = (0, 255, 0)
        img[i - 1, min(len(path),int(j)+1)] = (255, 0, 0)
        img[i - 1, min(len(path),int(j)+2)] = (0, 0, 255)

    img[-1][int(j)] = (0, 255, 0)
    img[-1][min(len(path),int(j)+1)] = (255, 0, 0)
    img[-1][min(len(path), int(j) + 2)] = (0, 0, 255)

    cv2.imwrite('outputs/path.jpg', img)
    plt.imshow(img)


for i in range(0, crop - 1):
    path_ = get_path(dir_, i)
    prova = mark('outputs/resized_colores.jpg', path_)

    plt.pause(0.05)
    plt.clf()
############################################################################
