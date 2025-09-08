import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#Data Rescaling
train_data = ImageDataGenerator(rescale=1./255)
val_data = ImageDataGenerator(rescale=1./255)

train_img = train_data.flow_from_directory(
    'dataset/train', 
    target_size=(64, 64), 
    batch_size=16, 
    class_mode='categorical')

val_img = val_data.flow_from_directory(
    'dataset/val', 
    target_size=(64, 64), 
    batch_size=16, 
    class_mode='categorical')

#Using CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax') #-> 2Classes(Cat, Dog)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Model Traing
model.fit(train_img, validation_data=val_img, epochs=10)

model.save('ImageClassifierModel.h5')
print("Model Saved!!!!!!")