import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
img_size = 128
batch_size = 32
num_classes = 26

data_dir = 'Dataset'

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to [0, 1]
    validation_split=0.2,
    # rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
)

# Data preprocessing for validation and testing data (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    data_dir + '/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# Load validation data
val_generator = val_test_datagen.flow_from_directory(
    data_dir + '/val',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # No need to shuffle for validation
)

# Load testing data
test_generator = val_test_datagen.flow_from_directory(
    data_dir + '/test',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # No need to shuffle for testing
)

# Define CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Evaluate the model on validation data
loss, accuracy = model.evaluate(val_generator, steps=val_generator.samples // batch_size)
print(f'Validation accuracy: {accuracy}')

# Evaluate the model on testing data
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Testing accuracy: {test_accuracy}')

# Save the model
model_path = 'Trained_Model/CNN_Model.keras'
model.save(model_path)

print(model.summary())
