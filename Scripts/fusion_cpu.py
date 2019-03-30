# -*- coding: utf-8 -*-
print 'script Init  ------ *************'
print 'Importando librerias...'
import numpy as np
from PIL import Image
import timeit
import sys, getopt
#import exifread
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import cv2

print '\tLibrerias importadas correctamente'

end = 0
start = 0
def getTranformada(test_image,diagonal):
	#multiplico cada fila por la diagonal
	test_image = test_image.astype(np.float32, copy=False)
	empty_list = np.zeros_like(test_image)
	#print '\tmultiplicando la matriz por la diagonal por primera vez'
	for x in test_image:
		x[:] = np.dot(x, diagonal)
	#print test_image
	test_image = test_image.transpose()
	#print 'tran'
	#print test_image
	#print '\tmultiplicando la matriz por la diagonal por segunda vez'
	for x in test_image:
		x[:] = np.dot(x, diagonal)
	#print 'mul2'
	#print test_image
	test_image = test_image.transpose()
	#print 'tran2'
	#print test_image
	#print '\tse finalizo la transformada exitosamente'
	return test_image


def getTranformada_Inversa(test_image,diagonal):
	#multiplico cada fila por la diagonal
	test_image = test_image.astype(np.float32, copy=False)
	empty_list = np.zeros_like(test_image)
	test_image = test_image.transpose()
	#print '\tmultiplicando la matriz por la diagonal por primera vez'
	for x in test_image:
		x[:] = np.dot(x, diagonal)
	test_image = test_image.transpose()
	#print '\tmultiplicando la matriz por la diagonal por segunda vez'
	for x in test_image:
		x[:] = np.dot(x, diagonal)
	
	#print '\tse finalizo la transformada exitosamente'
	return test_image


#Obtiene la matriz con 1/2 y -1/2
#inv:  Booleno que decide si se hace la diagonal inversa (con 1 y -1) (True) o normal (False)
def get_matriz_diagonal(V_mul, inv):
	shape_mul = V_mul.shape
	diagonal = np.zeros((shape_mul[0],shape_mul[0]))
	x=0
	y=0
	while y < shape_mul[1]-1:
		while x < shape_mul[0]-1:
			if inv:
				diagonal[x][y] = 1
				diagonal[x][y+(shape_mul[0])/2] = 1
			else:	
				diagonal[x][y] = 0.5
				diagonal[x][y+(shape_mul[0])/2] = 0.5
			if not x == shape_mul[0]-1:
				if inv:
					diagonal[x+1][y] = 1
					diagonal[x+1][y+(shape_mul[0])/2] = -1
				else:
					diagonal[x+1][y] = 0.5
					diagonal[x+1][y+(shape_mul[0])/2] = -0.5
				x=x+2
				break
		y=y+1
	if inv:
		return diagonal.transpose()
	else:
		return diagonal


def divide_componentes(transformada):
	mitad = transformada.shape[0]/2
	ca = transformada[0:mitad,0:mitad].copy()
	ch = transformada[0:mitad,mitad:].copy()
	cv = transformada[mitad:,0:mitad].copy()
	cd = transformada[mitad:,mitad:].copy()
	return ca, ch, cv, cd
#Obtiene los componenetes de la banda aplicandole la transformada
#banda: matriz a la cual se le aplica la transformada
#transformada: matriz con los valores para aplicar la transformada
#nivel: las veces que se le va a aplicar la transformada
def get_componentes(banda, diagonal, nivel):
	if nivel is not 0:
		#obtengo la transformada
		transformada = np.multiply(banda, diagonal)
		#optengo los componentes
		ca, ch, cv, cd = divide_componentes(transformada)
		#hago el siguiente nivel con los nuevos componentes
		diagonal_ca = get_matriz_transformada_inversa(ca, False)
		ca = get_componentes(ca, diagonal_ca, nivel-1)
		# uno los 4 componentes
		new_ca = np.concatenate((np.concatenate((ca, ch), axis=1), np.concatenate((cv, cd), axis=1)), axis=0)
		# les hago transformada inversa a la union de los componentes
		diagonal_inv_new_ca = get_matriz_transformada_inversa(new_ca, True)
		transformada_inv_new_ca = np.multiply(new_ca, diagonal_inv_new_ca)
		print '\t\ttransformada nivel %s aplicada' % (nivel)
		return transformada_inv_new_ca
	else:
		return banda


# Devuelve X, Y, tamaño de celda vertical y tamaño de celda horizontal de la imagen
def getXY(input_mul):   
	from osgeo import gdal      
	# print type(filetoread)    
	ds = gdal.Open(input_mul)    
	width = ds.RasterXSize    
	height = ds.RasterYSize    
	gt = ds.GetGeoTransform()
	band1 = ds.GetRasterBand(1).ReadAsArray()
	plt.imshow(band1, cmap='gist_earth')
	plt.show()
	minx = gt[0]
	miny = gt[3]    
	maxy = gt[3] + width * gt[4] + height * gt[5]    
	maxx = gt[0] + width * gt[1] + height * gt[2]    

	if not gt is None:
		return gt[0], gt[3], gt[1],gt[5]

def main():
	global end
	global start
	input_mul = sys.argv[1]
	input_pan = sys.argv[2]
	nivel = int(sys.argv[3])
	output_fus = sys.argv[4]
	#getXY(input_mul)
	# (1) Leer las imagenes
	print 'Leyendo las imagenes...'
	image_mul = cv2.imread(input_mul)
	
	#image_mul = Image.open(input_mul)
	#image_mul = np.array(image_mul)
	shape_mul = image_mul.shape
	#print image_mul
	if shape_mul[2]:
		print '\tla imagen multiespectral tiene '+str(shape_mul[2])+' bandas y tamaño '+str(shape_mul[0])+'x'+str(shape_mul[1])
	else:
		print '\tla primera imagen no es multiespectral'
	image_pan = Image.open(input_pan)
	#image_pan = cv2.imread(input_pan)
	#image_pan = cv2.cvtColor(image_pan, cv2.COLOR_BGR2GRAY)
	#image_pan = Image.open(input_pan)
	image_pan = np.array(image_pan)
	shape_pan = image_pan.shape
	print shape_pan
	print len(shape_pan)
	#print image_pan
	if len(shape_pan) == 2:
		print '\tla imagen Pancromatica tiene tamaño '+str(shape_pan[0])+'x'+str(shape_pan[1])
	else:
		print '\tla segunda imagen no es pancromatica'
	# (2) convertir la imagen a HSV
	print 'Conviertiendo RGB to HSV...'
	hsv_mul = colors.rgb_to_hsv(image_mul / 255.)
	print '\timagen convertida a HSV satisfactoriamente...'
	H_mul = hsv_mul[:,:,0]
	S_mul = hsv_mul[:,:,1]
	V_mul = hsv_mul[:,:,2]
	#print V_mul
	#plt.imsave('Test_data/01.hsv_mul.jpg',hsv_mul)
	#plt.imsave('Test_data/02.H.jpg',H_mul, cmap=plt.cm.gray)
	#plt.imsave('Test_data/03.S.jpg',S_mul, cmap=plt.cm.gray)
	#plt.imsave('Test_data/04.V.jpg',V_mul, cmap=plt.cm.gray)
	
	# transformada en el primer nivel al VALUE
	start = time.time()
	print 'Aplicando la transformada al VALUE...'
	diagonal = get_matriz_diagonal(V_mul, False) # False: para hacer la transformada normal
	#print 'diag'
	#print diagonal
	transformada_V_mul_image = getTranformada(V_mul,diagonal)
	#print 'transfor'
	#print transformada_V_mul_image
	#plt.imsave('Test_data/05.componentes_V_mul_image.jpg',transformada_V_mul_image, cmap=plt.cm.gray)
	
	# transformada en el primer nivel a la PANCROMATIC
	print 'Aplicando la transformada a la PANCROMATIC...'
	diagonal = get_matriz_diagonal(image_pan, False) # False: para hacer la transformada normal
	#print 'diag'
	#print diagonal
	transformada_pan_image = getTranformada(image_pan/ 255.,diagonal)
	#print 'trans'
	#print transformada_pan_image
	#plt.imsave('Test_data/06.componentes_pan_image.jpg',transformada_pan_image, cmap=plt.cm.gray)

	# obtengo los 4 componentes del VALUE y la PANCROMATIC (hago el segundo nivel) y los combino
	cap, chp, cvp, cdp = divide_componentes(transformada_pan_image)
	#print 'cap'
	#print cap
	#plt.imsave('Test_data/052.chp.jpg',chp, cmap=plt.cm.gray)
	#plt.imsave('Test_data/053.cvp.jpg',cvp, cmap=plt.cm.gray)
	#plt.imsave('Test_data/054.cdp.jpg',cdp, cmap=plt.cm.gray)
	cav, chv, cvv, cdv = divide_componentes(transformada_V_mul_image)
	#print 'cav'
	#print cav
	if nivel > 1:
		print 'Aplicando la transformada de segundo nivel a la PANCROMATIC...'
		#transformada del CAP optenidoe en el primer nivel
		diagonal = get_matriz_diagonal(cap, False) # False: para hacer la transformada normal
		#print 'diag'
		#print diagonal
		transformada_cap = getTranformada(cap,diagonal)
		#print 'tra'
		#print transformada_cap
		cap2, chp2, cvp2, cdp2 = divide_componentes(transformada_cap)
		#print 'cap2'
		#print cap2
		#print 'Aplicando la transformada de segundo nivel al VALUE...'
		#transformada del CAV optenidoe en el primer nivel
		diagonal = get_matriz_diagonal(cav, False) # False: para hacer la transformada normal
		#print 'diag'
		#print diagonal
		transformada_cav = getTranformada(cav,diagonal)
		#print 'tra'
		#print transformada_cav
		cav2, chv2, cvv2, cdv2 = divide_componentes(transformada_cav)
		#print 'cav2'
		#print cav2
		if nivel > 2:
			#transformada del CAP optenidoe en el primer nivel
			diagonal = get_matriz_diagonal(cap2, False) # False: para hacer la transformada normal
			#print 'diag'
			#print diagonal
			transformada_cap2 = getTranformada(cap2,diagonal)
			#print 'tra'
			#print transformada_cap2
			cap3, chp3, cvp3, cdp3 = divide_componentes(transformada_cap2)
			#print 'cap3'
			#print cap3
			print 'Aplicando la transformada de segundo nivel al VALUE...'
			#transformada del CAV optenidoe en el primer nivel
			diagonal = get_matriz_diagonal(cav2, False) # False: para hacer la transformada normal
			#print 'diag'
			#print diagonal
			transformada_cav2 = getTranformada(cav2,diagonal)
			#print 'tra'
			#print transformada_cav2
			cav3, chv3, cvv3, cdv3 = divide_componentes(transformada_cav2)
			#print 'cav3'
			#print cav3
			###__--- AQUI IRIA el Nivel 4 ---__###
			# combino los nuevos componentes del segundo nivel
			mitad = transformada_cap2.shape[0]/2
			transformada_cav2[0:mitad,mitad:] = chp3
			transformada_cav2[mitad:,0:mitad] = cvp3
			transformada_cav2[mitad:,mitad:] = cdp3
			cav3 = cav3*255.
			new_cav2_components = np.concatenate((np.concatenate((cav3, chp3), axis=1), np.concatenate((cvp3, cdp3), axis=1)), axis=0)
			#print 'mitad'
			#print mitad
			#print 'chp3'
			#print chp3
			#print 'cvp3'
			#print cvp3
			#print 'cdp3'
			#print cdp3
			#print 'cav3'
			#print cav3 
			#print 'new com'
			#print new_cav2_components
			
			#plt.imsave('Test_data/new_cav.jpg',new_cav_components, cmap=plt.cm.gray)
			#Transformada inversa al tercer nivel
			diagonal_inv = get_matriz_diagonal(new_cav2_components, True)
			#print 'dia'
			#print diagonal_inv
		
			cav2 = getTranformada_Inversa(new_cav2_components,diagonal_inv)
			#print 'tra'
			#print cav2
			cav2 = cav2/255.
			#print 'cav2'
			#print cav2
		# combino los nuevos componentes del segundo nivel
		mitad = transformada_cap.shape[0]/2
		transformada_cav[0:mitad,mitad:] = chp2
		transformada_cav[mitad:,0:mitad] = cvp2
		transformada_cav[mitad:,mitad:] = cdp2
		cav2 = cav2*255.
		new_cav_components = np.concatenate((np.concatenate((cav2, chp2), axis=1), np.concatenate((cvp2, cdp2), axis=1)), axis=0)
		#plt.imsave('Test_data/new_cav.jpg',new_cav_components, cmap=plt.cm.gray)
		#Transformada inversa al segundo nivel
		diagonal_inv = get_matriz_diagonal(new_cav_components, True)
		cav = getTranformada_Inversa(new_cav_components,diagonal_inv)
		cav = cav/255.
		#plt.imsave('Test_data/new_cav_nivel2.jpg',cav, cmap=plt.cm.gray)
	#Combino los componentes resultantes
	print 'Combinando los componentes de VALUE y PANCROMATIC...'
	mitad = transformada_pan_image.shape[0]/2
	transformada_V_mul_image[0:mitad,mitad:] = chp
	transformada_V_mul_image[mitad:,0:mitad] = cvp
	transformada_V_mul_image[mitad:,mitad:] = cdp
	cav = cav*255.
	new_V_mul_components = np.concatenate((np.concatenate((cav, chp), axis=1), np.concatenate((cvp, cdp), axis=1)), axis=0)
	#plt.imsave('Test_data/07.componentes_new_V_mul.jpg',transformada_V_mul_image, cmap=plt.cm.gray)
	print '\tcomponentes combinados exitosamente'

	# Le aplico la transformada inversa a los componentes del nuevo nuevo V_mul
	print 'Aplicando la transformada inversa a los componentes del nuevo VALUE...'
	diagonal_inv = get_matriz_diagonal(V_mul, True)
	new_V_mul = getTranformada_Inversa(new_V_mul_components,diagonal_inv)
	#plt.imsave('Test_data/08.new_V_mul.jpg',new_V_mul, cmap=plt.cm.gray)
	print '\ttransformada aplicada exitosamente'
	end = time.time()
	# Obtengo la nueva imagen RGB
	print 'Obteniendo la nueva imagen RGB...'
	print '\tuniendo H, S y new_V'
	old_RGB = colors.hsv_to_rgb(hsv_mul)
	#plt.imsave('Test_data/old_RGB.jpg',old_RGB)
	hsv_mul[:,:,2] = new_V_mul/255.
	#plt.imsave('Test_data/new_hsv_mul.jpg',hsv_mul)
	print '\tconviertiendo HSV a RGB'
	new_RGB = colors.hsv_to_rgb(hsv_mul)
	#print 'new RGB'
	#print new_RGB
	print '\tguardando nueva RGB'
	#plt.imsave('Test_data/09.new_RGB.jpg',new_RGB)
	plt.imsave(output_fus+'.tif',new_RGB)
	#plt.imsave(output_fus,new_RGB)
	plt.imsave('/home/nvera/andres/'+output_fus[5:]+'.tif.png',new_RGB)
	print '************* ------ Script Exit  ------ *************'



main()
tiempo = (end - start)
print tiempo
