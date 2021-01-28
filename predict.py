from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix
import numpy as np
import time

# -----------------------------------------------------------------------------
# Dimensions of images
# -----------------------------------------------------------------------------
img_width, img_height = 200, 200
input_shape = (img_width, img_height, 3)

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
test_model = load_model('models/VGG16/1.h5')
#test_model.load_weights('weights.h5')
#test_model.summary()

# -----------------------------------------------------------------------------
# Image data path
# -----------------------------------------------------------------------------
testdir = "data/test/"

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------

num_of_test_samples=270
batch_size=1

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    testdir,
    target_size=(img_width, img_height),
    class_mode='categorical',
    shuffle=False,
    batch_size=batch_size)

start=time.time()
Y_pred = test_model.predict_generator(test_generator,num_of_test_samples // batch_size)
end=time.time()
duration=end-start
print("Predict time: ",duration)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)

print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))