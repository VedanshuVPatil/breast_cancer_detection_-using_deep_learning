import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = data_gen.flow_from_directory(
    "dataset/",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    "dataset/",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))


for layer in base_model.layers[-5:]:  
    layer.trainable = True


model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=5)

#To Save Model
model.save("breast_cancer_vgg19_finetuned.h5")


