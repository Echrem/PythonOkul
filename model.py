import numpy as np
import tensorflow as tf
import cv2

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model("sign_language_model.h5")

# Sınıf isimlerini buraya ekleyin
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Kamera aç
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aynalı görüntüyü düzelt
    frame = cv2.flip(frame, 1)

    # Model için görüntüyü yeniden boyutlandır
    resized_frame = cv2.resize(frame, (640, 480))
    normalized_frame = resized_frame / 255.0  # Normalizasyon
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Batch boyutunu ekleyin

    # Tahmin yap
    predictions = model.predict(input_frame)
    predicted_class = class_names[np.argmax(predictions)]  # Tahmin edilen sınıfı al

    # Tahmini ekrana yazdır
    cv2.putText(frame, f'Prediction: {predicted_class}', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Ekranda görüntüyü göster
    cv2.imshow('Sign Language Recognition', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
