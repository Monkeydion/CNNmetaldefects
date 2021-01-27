from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, DenseNet121, DenseNet169, DenseNet201, Xception 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 200, 200

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
epochs = 100
batch_size = 16

input_shape = (img_width, img_height, 3)

#The Model (Creation)
model=VGG16(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=VGG19(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=ResNet50(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=ResNet101(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=ResNet152(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=InceptionV3(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=InceptionResNetV2(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=DenseNet121(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=DenseNet169(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=DenseNet201(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)
#model=Xception(include_top=True,weights=None,input_tensor=None,input_shape=(img_height, img_width ,3),pooling='max',classes=6)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(1e-3),
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=10
    )

# this is the augmentation configuration we will use for testing: only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# creating the training flow object
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

print("train_generator")

# creating the validation flow object
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

print("validation_generator")

# callbacks are used to save weights for each epoch
# IMPORTANT: change directory everytime model is changed
callbacks = [
    ModelCheckpoint("models/VGG16/{epoch}.h5"),
]

#optimizes the model to accurately predict the classes
history=model.fit_generator(
    train_generator,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator)

# to visualize the training history
plt.figure(figsize=(18,18))

plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()