#load packages
library(tensorflow)
library(keras)
library(EBImage)

#mnist data
c(c(trainx, trainy), c(testx, testy)) %<-% dataset_mnist()
#only use the training data images and use value of 3
trainx<-trainx[trainy==3,,] 


#reshape images
trainx<- array_reshape(trainx,c(nrow(trainx),28,28,1))

#normalize training data so that range is between 0 and 1
trainx<-trainx/255

#create generator network
#specify dimensions of images
h<-28;w<-28;c<-1;l=28 #l is for latent dimensions
gi<-layer_input(shape=l)
go<- gi %>% layer_dense(units=32*14*14)%>%
  layer_activation_leaky_relu()%>%
  layer_reshape(target_shape=c(14,14,32))%>%
  #add convolution layer
  layer_conv_2d(filters=32,
                kernel_size=5,
                padding="same")%>%
  layer_activation_leaky_relu()%>%
  layer_conv_2d_transpose(filter=32,
                          kernel_size = 4,
                          stride=2,
                          padding="same")%>%
  layer_activation_leaky_relu()%>%
  layer_conv_2d(filters=1,
                kernel_size = 5,
                activation = "tanh",
                padding="same")
g<-keras_model(gi,go)
summary(g)


#create discriminator network
di<- layer_input(shape=c(h,w,c))
do<- di%>%
  layer_conv_2d(filters=64,
                kernel_size=4)%>%
  layer_activation_leaky_relu()%>%
  layer_flatten()%>%
  layer_dropout(rate=0.3)%>%
  #create dense layer for classification 
  layer_dense(units=1,
              activation="sigmoid")
d<-keras_model(di,do)
summary(d)

#compile the discriminator network
d%>%compile(optimizer='rmsprop',
            loss='binary_crossentropy')

#freeze weights in discriminator and compile
freeze_weights(d)
gani<-layer_input(shape=l)
gano<-gani%>%g%>%d
gan<-keras_model(gani,gano)
gan%>%compile(optimizer='rmsprop',
              loss="binary_crossentropy")

summary(gan)


#####################################
#train
b<-50
#create directory for storage of images
dir<-"gan3_img"
dir.create(dir)

#initialize discriminator loss and generator loss
start<-1;dloss<-NULL;gloss<-NULL

#step 1. Generate 50 fake images using random normal distribution
for (i in 1:100){noise<-matrix(rnorm(b*l),
                               nrow=b,
                               ncol=l)
fake<-g%>%predict(noise)

#step 2 combine real and fake images
stop<-start+b-1

real<- trainx[start:stop,,,]
real<-array_reshape(real,c(nrow(real),28,28,1))
rows<- nrow(real)
both<-array(0,dim=c(rows*2,dim(real)[-1]))
both[1:rows,,,]<-fake
both[(rows+1):(rows*2),,,]<-real
#bind fake and real 
#use range between 0.9 and 1(instead of 1) for fake image
#use range between 0 and 0.1(instead of 0)for real image
labels<-rbind(matrix(runif(b,0.9,1),
                     nrow=b,
                     ncol=1),
              matrix(runif(b,0,0.1),
                     nrow=b,
                     ncol=1))


start<-start+b


#step 3 train the discriminator
dloss[i]<-d%>%train_on_batch(both,labels)

#step 4 train generator using GAN
fakeAsReal<-array(runif(b,0,0.1),dim=c(b,1))
gloss[i]<-gan%>%train_on_batch(noise,fakeAsReal)

#step 5 save fake images
f<-fake[1,,,]
dim(f)<-c(28,28,1)
image_array_save(f,path=file.path(dir,
                          paste0("f",i ,".png")))}

#########

temp=list.files(pattern='*.png')
mypic<-list()
for (i in 1:length(temp)){mypic[[i]]<-readImage(temp[[i]])}
par(mfrow=c(10,10))
for(i in 1:length(temp))plot(mypic[[i]])
