import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2

from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
from matplotlib import cm


#dimension required for image
dim=(150,150)
#input image
image=imread('763.jpg')
#convert to gray scale image
image=gray2rgb(image)
#resize the image and preprocess
image=resize(image,dim)
image=image[:,:,1]
image=image.astype('float32')

#Labels to use for subplots
labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

# select the Kernel size to an odd value
kSize = 11
#select the number of filters
nFilters=16

#create the array of parameters such as Lambda, Theta, Psi, Sigma, Gamma
lambdaAr = np.array([kSize-2] * nFilters) #contant
thetaAr = np.arange(0, 360+1,(360+1)/(nFilters*1.)) #changing
thetaAr=thetaAr * np.pi / 180.
psiAr = np.arange(0, 360+1,((360+1))/(nFilters*1.))
psiAr = psiAr * np.pi / 180

sigmaAr = np.arange(kSize/2, ((kSize*2)/3+1), (((kSize*2)/3+1)-(kSize/2))/(nFilters*1.))
gammaAr = np.ones(nFilters)


#3D display of Gabor response
def GaborRespDisp3D(ha,matScores,m,n,zLabel):
    nx, ny = m,n
    x = range(nx)
    y = range(ny)    
    X, Y = np.meshgrid(x, y)  
    ha.plot_surface(X, Y, matScores[0:m,0:n],cmap=cm.coolwarm)
    ha.contour(X, Y, matScores[0:m,0:n])
    ha.text(85, -70, 0,'('+labels[count-1]+')',color='black',fontsize=8,fontweight='bold')
    ha.set_zlim(0, 25)
    ha.xaxis.set_tick_params(labelsize=8)
    ha.yaxis.set_tick_params(labelsize=8)
    ha.zaxis.set_tick_params(labelsize=8)
    ha.set_xlabel('X axis',fontsize=8)
    ha.set_ylabel('Y axis',fontsize=8)
    ha.set_zlabel(zLabel,fontsize=8)
    custom_ticks = np.linspace(0, 25, 4, dtype=float)
    ha.set_zticks(custom_ticks)

#3D display of Gabor filters    
def GaborFilterDisp3D(ha,matScores,m,n,zLabel):
    nx, ny = m,n
    x = range(nx)
    y = range(ny)    
    X, Y = np.meshgrid(x, y)  
    ha.plot_surface(X, Y, matScores[0:m,0:n],cmap=cm.coolwarm)
    ha.contour(X, Y, matScores[0:m,0:n])
    ha.text(10, -9, 0,'('+labels[count-1]+')',color='black',fontsize=8,fontweight='bold')
    ha.set_xlim(0, 11)
    ha.set_ylim(0, 11)
    ha.set_zlim(-0.5, 1)
    ha.xaxis.set_tick_params(labelsize=8)
    ha.yaxis.set_tick_params(labelsize=8)
    ha.zaxis.set_tick_params(labelsize=8,length=2)    
    ha.set_xlabel('X axis',fontsize=8)
    ha.set_ylabel('Y axis',fontsize=8)
    ha.set_zlabel(zLabel,fontsize=8)
    custom_ticks = np.linspace(-0.5, 1, 3, dtype=float)
    ha.set_zticks(custom_ticks)

#create a figure with tight layout
hf=plt.figure()
hf.tight_layout()
count =1
for i in range(0,16,1):
    print(i)
    ha=hf.add_subplot(4,4,count,projection='3d')
    g_theta=thetaAr[i]
    g_lambda=lambdaAr[i]
    g_psi=psiAr[i]
    g_sigma=sigmaAr[i]
    g_gamma=gammaAr[i]
    #create Gabor kernel with above parameters
    kernel = cv2.getGaborKernel((kSize, kSize), g_sigma,g_theta,g_lambda,g_gamma,g_psi)    
    #display gabor filter
    GaborFilterDisp3D(ha,kernel ,kernel.shape[0],kernel.shape[1],zLabel='g(x,y)')
    count=count+1
plt.show()

#create a figure with tight layout
hf=plt.figure()
hf.tight_layout()
count =1
for i in range(0,16,1):
    print(i)
    ha=hf.add_subplot(4,4,count,projection='3d')
    g_theta=thetaAr[i]
    g_lambda=lambdaAr[i]
    g_psi=psiAr[i]
    g_sigma=sigmaAr[i]
    g_gamma=gammaAr[i] 
    #create Gabor kernel with above parameters
    kernel = cv2.getGaborKernel((kSize, kSize), g_sigma,g_theta,g_lambda,g_gamma,g_psi)
    #perform the 2D convolution
    filtered_img = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    #display gabor response
    GaborRespDisp3D(ha,filtered_img ,filtered_img.shape[0],filtered_img.shape[1],'g-resp')
    count=count+1
plt.show()