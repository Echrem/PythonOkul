import tensorflow as tf
from tensorflow import keras
import os

# Veri setinin bulunduğu klasörün yolu
DATASET_PATH = "/home/kali/Desktop/PythonOkul/data"  # Klasörün yolu

# Veri setini yükle ve eğitim/test olarak böl
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(640, 480),  # Görüntü boyutlandırma
    batch_size=2,         # Batch boyutu
    validation_split=0.2,  # %20'si dogrulamak icin
    subset="training",
    seed=123               # Rastgelelik için seed
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(224, 224),
    batch_size=2,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Performans için veri önbelleğe al
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Burada, class_names'ı doğrudan almak yerine, dataset'in `class_names`'ına bakmaya çalışacağız
class_names = train_dataset.class_names if hasattr(train_dataset, 'class_names') else sorted(os.listdir(DATASET_PATH))
print("Sınıflar:", class_names)

# Model oluştur
model = keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(640, 480, 3)),  # Normalizasyon
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Çıktı katmanı
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10  # Epoch sayısı
)

# Eğitimi kaydet
model.save("sign_language_model.h5")
print("Model kaydedildi!")
