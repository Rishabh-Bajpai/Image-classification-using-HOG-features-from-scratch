import numpy as np
from PIL import Image

def preprocess(filename):
	imageInputOpen=Image.open(filename).convert('L')
	imageInput = np.array(imageInputOpen,dtype=np.int16)
	print("File opened")
	[sizeY,sizeX]=imageInput.shape
	imagePre=imageInput[1:sizeY:2,1:sizeX:2]
	[sizeY,sizeX]=imagePre.shape
	imagePre=imagePre*(255/np.amax(imagePre))

	return imagePre


def cal_gradiant(imagePre):
	[sizeY,sizeX]=imagePre.shape
	gx=np.zeros((sizeY,sizeX))
	for xi in range(1,sizeX-1):
		gx[:,xi]=(imagePre[:,xi+1]-imagePre[:,xi-1])
	gy=np.zeros((sizeY,sizeX))
	for yi in range(1,sizeY-1):
		gy[yi,:]=(imagePre[yi+1,:]-imagePre[yi-1,:])
	g=np.sqrt(gx*gx+gy*gy)
	gx[gx==0]=0.00001;
	g_angle=np.arctan(gy/gx)*180/3.14;
	
	#Uncomment to see the gradiant
	#imagePreFile=Image.fromarray(g.astype(np.uint8))
	#imagePreFile.show()
	return g,g_angle

def calc_histPos(value,value_grad,histogram_cell):
	L=np.int(np.floor(value/20.0))
	U=L+1
	if U==9:
		U=0
	pl=(value-L*20)/20
	pu=1-pl
	histogram_cell[L]=pl*value_grad+histogram_cell[L]
	histogram_cell[U]=pu*value_grad+histogram_cell[U]
	return histogram_cell


def calc_histogram(g,ga):
	g=np.reshape(g,(64,1))
	ga=np.reshape(ga,64)
	ga[ga<0]=ga[ga<0]+180
	#print(ga)
	histogram_cell=np.zeros(9)
	for i in range(64):
		histogram_cell=calc_histPos(ga[i],g[i],histogram_cell)
	return histogram_cell
	

	
def hist_gradiant(gradiant,gradiant_angle):
	[sizeY,sizeX]=gradiant.shape
	histogram_cell_mat=np.zeros((sizeY // 8,sizeX // 8,9))
	for yc in range(sizeY // 8):
		for xc in range(sizeX // 8):
			histogram_cell_mat[yc,xc,0:9]=calc_histogram(gradiant[yc*8:yc*8+8,xc*8:xc*8+8],gradiant_angle[yc*8:yc*8+8,xc*8:xc*8+8])
	return histogram_cell_mat

def normalized_hist(histogram_block):
	histogram_block1=np.reshape(histogram_block,(36))
	histogram_block_norm=histogram_block/np.sqrt(np.sum(np.square(histogram_block1)))
	return histogram_block_norm
def hist_vector(histogram_block):
	histogram_block_norm=np.reshape(histogram_block,(36))
	return histogram_block_norm

def Block_norm(histogram_block_mat):
	histogram_vector_mat=np.zeros((15,15,36))
	for yi in range(15):
		for xi in range(15):
			histogram_block_mat[yi:yi+2,xi:xi+2,:]=normalized_hist(histogram_block_mat[yi:yi+2,xi:xi+2,:])
			histogram_vector_mat[yi:yi+2,xi:xi+2,:]=hist_vector(histogram_block_mat[yi:yi+2,xi:xi+2,:])


	return histogram_vector_mat


def hog(filename):
	imagePre=preprocess(filename)
	gradiant,gradiant_angle=cal_gradiant(imagePre)
	histogram_block_mat=hist_gradiant(gradiant,gradiant_angle)
	histogram_vector_mat=Block_norm(histogram_block_mat)
	feature_vect=np.reshape(histogram_vector_mat,(8100))
	return feature_vect