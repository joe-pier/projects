import cv2
import pandas as pd



prova = cv2.imread('monnalisa.jpg', cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(prova, cv2.COLOR_BGR2GRAY)
pandas = pd.DataFrame(image)

pandas.to_csv('miao.csv')
print(pandas)


