import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import pathlib
import random
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

path_to_food = pathlib.Path('../pizza_steak')


def _view_random_image(food: str):
    target_folder = path_to_food.joinpath('train/' + food)
    print(target_folder)
    random_image = random.sample(os.listdir(target_folder), 1)
    target_folder = target_folder.joinpath(random_image[0])
    img = mpimg.imread(target_folder)
    plt.imshow(img)
    plt.axis("off");
    plt.show()
    print(f"Image shape: {img.shape}")  # show the shape of the image


def _first_cnn_model():
    train_dir = "../pizza_steak/train"
    test_dir = "../pizza_steak/test"

    train_data_generator = ImageDataGenerator(rescale=1. / 255)
    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    # Importovanje slika i podala i  delove od 30
    train_data = train_data_generator.flow_from_directory(directory=train_dir,
                                                          batch_size=30,  # delovi od 30
                                                          target_size=(224, 224),  # size slika
                                                          shuffle=True,  # izmesati ih
                                                          class_mode="binary",  # binarna jer radimo sa 0 i jedinicama
                                                          seed=42)

    test_data = test_data_generator.flow_from_directory(directory=test_dir,
                                                        batch_size=30,
                                                        target_size=(224, 224),
                                                        shuffle=True,
                                                        class_mode="binary",
                                                        seed=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=10,
                               kernel_size=3,
                               activation="relu",
                               input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2,
                                  padding="valid"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"])

    history = model.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))


def _second_model():
    train_dir = "../pizza_steak/train"
    test_dir = "../pizza_steak/test"

    train_data_generator = ImageDataGenerator(rescale=1. / 255)
    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    # Importovanje slika i podala i  delove od 30
    train_data = train_data_generator.flow_from_directory(directory=train_dir,
                                                          batch_size=30,  # delovi od 30
                                                          target_size=(224, 224),  # size slika
                                                          shuffle=True,  # izmesati ih
                                                          class_mode="binary",  # binarna jer radimo sa 0 i jedinicama
                                                          seed=42)

    test_data = test_data_generator.flow_from_directory(directory=test_dir,
                                                        batch_size=30,
                                                        target_size=(224, 224),
                                                        shuffle=True,
                                                        class_mode="binary",
                                                        seed=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=10,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               activation="relu",
                               input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.optimizers.Adam(),
                  metrics=["accuracy"])

    history = model.fit(train_data, epochs=5, steps_per_epoch=len(train_data), validation_data=test_data,
                        validation_steps=len(test_data))

    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.show()


def _third_model():
    train_dir = "../pizza_steak/train"
    test_dir = "../pizza_steak/test"

    train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              rotation_range=20,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              horizontal_flip=True)
    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    # Importovanje slika i podala i  delove od 30
    train_data = train_data_generator.flow_from_directory(directory=train_dir,
                                                          batch_size=30,  # delovi od 30
                                                          target_size=(224, 224),  # size slika
                                                          shuffle=False,  # izmesati ih
                                                          class_mode="binary",  # binarna jer radimo sa 0 i jedinicama
                                                          seed=42)

    test_data = test_data_generator.flow_from_directory(directory=test_dir,
                                                        batch_size=30,
                                                        target_size=(224, 224),
                                                        shuffle=False,
                                                        class_mode="binary",
                                                        seed=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=10,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               activation="relu",
                               input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.optimizers.Adam(),
                  metrics=["accuracy"])

    history = model.fit(train_data, epochs=5, steps_per_epoch=len(train_data), validation_data=test_data,
                        validation_steps=len(test_data))

    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.show()


def _fourth_model():
    train_dir = "../pizza_steak/train"
    test_dir = "../pizza_steak/test"

    train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              rotation_range=20,
                                              shear_range=0.2,
                                              zoom_range=0.1,
                                              )
    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    # Importovanje slika i podala i  delove od 30
    train_data = train_data_generator.flow_from_directory(directory=train_dir,
                                                          batch_size=30,  # delovi od 30
                                                          target_size=(224, 224),  # size slika
                                                          shuffle=False,  # izmesati ih
                                                          class_mode="binary",  # binarna jer radimo sa pizza i steak
                                                          seed=42)

    test_data = test_data_generator.flow_from_directory(directory=test_dir,
                                                        batch_size=30,
                                                        target_size=(224, 224),
                                                        shuffle=False,
                                                        class_mode="binary",
                                                        seed=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=10,
                               kernel_size=2,
                               activation="relu",
                               input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(10, 2, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 2, activation="relu"),
        tf.keras.layers.Conv2D(10, 2, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer=tf.optimizers.Adam(),
                  metrics=["accuracy"])

    history = model.fit(train_data, epochs=5, steps_per_epoch=len(train_data), validation_data=test_data,
                        validation_steps=len(test_data))

    pd.DataFrame(history.history).plot(figsize=(10, 7))
    plt.show()


# TODO napomene za dosada odradjen posao
# Overfitting kada se model ponasa dobro tj daje dobra predvidjanja za trening podatka, a gresi dosta na realnim podacima
# Da bi se izbegao overfitting potrebno je kada se radi sa slikama da se slima malo uoblici
# da se siftuje malo po visini ili sirini, da se zumbira, da se rotira malo, da se izmiksa shuffle
# to se radi da bi algoritam neurnosnke mreze pravio paterne i da bi kasnije lakse prepoznavao
# batch size da se podeli po paketima jer nije bas u mogucnosti da sve slike stavi u RAM
# takodje kod binarne class mode stavljamo da je binary jer imamo samo neka 2 modela koja treba da prediktujemo
# ovde se stavlja categorical jer imamo multi class klasifikaciju
# Sta predstavlja konvolucionalna neuronska mreza
# Da bi se slike mogle obraditi jeste da su one zapisane kao matrica sa brojevima od 0-255
# Lakse je da stavimo neki ulazni parametar sto je Con2D on slike moze da ima na 3 kanala
# RGB kanali svaka slika je presek ta tri kanala i tako se dobijaju boje
# Con2D sluzi da bi slike pretrazila po kanalima i da bi napravila neke paterne
# neke linije prepoznatiljive oblike npr i to se pretvara u con2d
# parametri kernel to je koliko ce matrica da se pravi za pretragu piksela
# filter kako ce se pretrazivati, padding da li ce se dodati na sliku jos koji sloj da bude veca matrica
# funkcija aktivacije kako ce se racunati,kada pronadjemo paterne nije lose da malo skaliramo jer necemo stalno povecavati
# matricu, kada npr sa 9x9 matricu sa Pool mi mozemo samo bitne delove matrice da uzmemo, ulazni podaci su matrice koje imaju oznacene
# neke delove koje je algoritam pronasao kao prepoznatljive delove, koje ce mu kasnije koristiti za prepoznavanje.
# Mozemo ova 2 dela da ponovimo vise puta. Nakon toga koristimo flattern da npr ulazne matrice koje su oblika 5x5x2 stavimo u jedan vektor
# koji ima 50 elemenata. I na kraju Dense sa fjom softmax jer ta funkcija daje neku manju verovatnocu da ce zaokruziti na neki prepoznatljivi objekat
# tipa 0.7 su cipele itd. categorical_crossentropy loss fja za multi class kategorije binary_crossentropy  za binarne :D

def _multi_class_classification():
    train_dir = "10_food_classes_all_data/train/"
    test_dir = "10_food_classes_all_data/test/"

    train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              rotation_range=20,  # note: this is an int not a float
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True)

    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    train_data = train_data_generator.flow_from_directory(train_dir, shuffle=True, batch_size=32,
                                                          target_size=(224, 224),
                                                          class_mode="categorical")
    test_data = test_data_generator.flow_from_directory(test_dir, shuffle=True, batch_size=32,
                                                        target_size=(224, 224),
                                                        class_mode="categorical")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(10, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    history = model.fit(train_data, epochs=6, steps_per_epoch=len(train_data), validation_data=test_data,
                        validation_steps=len(test_data))

    model.save('_lecutre03')

if __name__ == "__main__":
    _multi_class_classification()
