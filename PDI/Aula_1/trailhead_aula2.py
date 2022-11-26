# Importação das bibliotecas
import cv2
import numpy as np
# Leitura da imagem com a função imread()
imagem = cv2.imread('entrada.jpg')
print('Largura em pixels: ', end='')
print(imagem.shape[1])
#largura da imagem print('Altura em pixels: ', end='')
print(imagem.shape[0])
#altura da imagem
print('Qtde de canais: ', end='')
print(imagem.shape[2])
# Mostra a imagem com a função imshow
cv2.imshow("Nome da janela", imagem)
cv2.waitKey(0) #espera pressionar qualquer tecla
# Salvar a imagem no disco com função imwrite()
cv2.imwrite("saida.jpg", imagem)

imagem2 = cv2.imread('ponte.jpg') 
(b, g, r) = imagem2[0, 0] #veja que a ordem BGR e não RGB 
print('O pixel (0, 0) tem as seguintes cores:') 
print('Vermelho:', r, 'Verde:', g, 'Azul:', b)  

imagem3 = cv2.imread('ponte.jpg') 
for y in range(0, imagem3.shape[0]):
   for x in range(0, imagem3.shape[1]):
     imagem[y, x] = (255,0,0)
cv2.imshow("Imagem modificada", imagem3) 

imagem4 = cv2.imread('ponte.jpg')
recorte = imagem4[100:200, 100:200]
cv2.imshow("Recorte da imagem", recorte)
cv2.imwrite("recorte.jpg", recorte) #salva no disco

img = cv2.imread('ponte.jpg')
cv2.imshow("Original", img)
largura = img.shape[1]
altura = img.shape[0]
proporcao = float(altura/largura)
largura_nova = 320 #em pixels
altura_nova = int(largura_nova*proporcao)
tamanho_novo = (largura_nova, altura_nova)
img_redimensionada = cv2.resize(img,
tamanho_novo, interpolation = cv2.INTER_AREA)
cv2.imshow('Resultado', img_redimensionada)
cv2.waitKey(0) 

cv2.imshow("Original", img) 
flip_horizontal = img[::-1,:]#comando equivalente abaixo 
#flip_horizontal = cv2.flip(img, 1)  
cv2.imshow("Flip Horizontal", flip_horizontal) 
flip_vertical = img[:,::-1] #comando equivalente abaixo 
#flip_vertical = cv2.flip(img, 0)  
cv2.imshow("Flip Vertical", flip_vertical) 
flip_hv = img[::-1,::-1] #comando equivalente abaixo  
#flip_hv = cv2.flip(img, -1) 
cv2.imshow("Flip Horizontal e Vertical", flip_hv) 
cv2.waitKey(0) 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)
cv2.waitKey(0) 

(canalAzul, canalVerde, canalVermelho) = cv2.split(img)
cv2.imshow("Vermelho", canalVermelho) 
cv2.imshow("Verde", canalVerde) 
cv2.imshow("Azul", canalAzul) 
cv2.waitKey(0) 

(canalAzul, canalVerde, canalVermelho) = cv2.split(img)  
zeros = np.zeros(img.shape[:2], dtype = "uint8")  
cv2.imshow("Vermelho", cv2.merge([zeros, zeros, canalVermelho]))  
cv2.imshow("Verde", cv2.merge([zeros, canalVerde, zeros])) 
cv2.imshow("Azul", cv2.merge([canalAzul, zeros, zeros])) 
cv2.imshow("Original", img) 
cv2.waitKey(0) 

