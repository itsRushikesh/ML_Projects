# ML_Projects

To build an end-to-end CNN-based image classification system in Python for categorizing bottles as Type-I, Type-II, or Uncategorized, you'll need the following steps:
🧠 Workflow Overview
1. Organize Data – Prepare your training data in folders.
2. Preprocess Data – Load and preprocess using ImageDataGenerator.
3. Build CNN Model – Use TensorFlow/Keras to define and train a model.
4. Train Model – Train the model on your image dataset.
5. Classify Input from Camera – Use OpenCV to capture an image and classify.
6. Label 'Uncategorized' – If the confidence is below a threshold or label is unknown.
📁 Assumed Directory Structure
Organize your images like this:

graphqlCopyEditbottle_data/
│
├── train/
│   ├── Type-I/
│   └── Type-II/
│
└── validation/
    ├── Type-I/
    └── Type-II/

Any other object will be tested during live prediction. If it's not confidently Type-I or Type-II, we label it "Uncategorized".

✅ Full End-to-End Code

pythonCopyEditimport osimport numpy as npimport cv2from tensorflow.keras.models import Sequentialfrom tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropoutfrom tensorflow.keras.preprocessing.image import ImageDataGeneratorfrom tensorflow.keras.models import load_model
# ========== Step 1: Data Preparation ==========img_width, img_height = 150, 150train_data_dir = 'bottle_data/train'validation_data_dir = 'bottle_data/validation'batch_size = 16num_classes = 2  # Type-I and Type-II
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
# ========== Step 2: Build the CNN Model ==========model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# ========== Step 3: Train the Model ==========model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)
# Save the modelmodel.save('bottle_classifier.h5')
# ========== Step 4: Live Camera Prediction ========== def classify_image(frame, model, threshold=0.7):
    resized = cv2.resize(frame, (img_width, img_height))
    img_array = np.expand_dims(resized / 255.0, axis=0)
    prediction = model.predict(img_array)[0]
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    if confidence < threshold:
        return "Uncategorized", confidence
    else:
        return class_indices[predicted_class], confidence
# Load modelmodel = load_model('bottle_classifier.h5')
# Start webcamcap = cv2.VideoCapture(0)print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    label, conf = classify_image(frame, model)
    cv2.putText(frame, f"{label} ({conf*100:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Bottle Classifier', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

🛠️ Notes:
• The model currently uses a basic CNN. For better accuracy, consider using a pretrained model like MobileNetV2 with transfer learning.
• The threshold value defines how confident the model should be before it declares a result.
• This solution handles live camera classification and includes the "Uncategorized" logic based on confidence
