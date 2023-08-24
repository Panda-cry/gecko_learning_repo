import tensorflow as tf


def get_images():
    IMG_SIZE = (224, 224)
    train_dir = "10_food_classes_10_percent/train/"
    test_dir = "10_food_classes_10_percent/test/"

    # train_dir = "10_food_classes_1_percent/train/"
    # test_dir = "10_food_classes_1_percent/test/"

    train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                     image_size=IMG_SIZE,
                                                                     label_mode="categorical",
                                                                     batch_size=32)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                     image_size=IMG_SIZE,
                                                                     label_mode="categorical",
                                                                     #batch_size=32
                                                                    )

    return  train_data, test_data


def create_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
        tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
        tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
        # preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNetB0
    ], name="data_augmentation")
    input_shape = (224, 224, 3)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    inputs = tf.keras.layers.Input(shape=input_shape,name="input_layer")
    x = data_augmentation(inputs)
    x = base_model(x,training=False)

    x  = tf.keras.layers.GlobalAveragePooling2D()(x)

    outputs = tf.keras.layers.Dense(10,activation="softmax")(x)

    model = tf.keras.Model(inputs,outputs)
    base_model.trainable = True

    # Freeze all layers except for the
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",
                  metrics=["accuracy"])

    train_data, test_data = get_images()
    history = model.fit(train_data,epochs=5,steps_per_epoch=len(train_data),
                        validation_data=test_data,validation_steps=len(test_data))


#aktivaciona funkcija je ulazni parametar koji se ubacuje u f(x) i kasnije kao izlaz vidimo da li je neuron aktivan ili ne
#dosta problema imaju aktiacione fje tako da treba paziti kod reggresion treba da se koriste linearne fje
#za binarj  classification sigmoid kao za izlaz, classification softmax, a mutilabel classfication sigmoid
#sigmoid i sofrmax, than se vise koriste za output da zaokruze te vrednosti izmedju 0,1 -2,2 dok se u hidden layers koriste relu
# za pocetak ali mozemo da koristimo i malo unapredjenije metode relu ima ih dosta,
#tune data je kada zelimo tj mozemo da model koji preuzmemo sa tenserflow hub da ubacime neke nase stavri
# mozemo da ulazne podatke malo izmanjamo data- argumentation  da krositimo pre ulaznog sloja,takodje da menjamo output sloj
# sada model sa tenserflow hub moze biti frozen i unfrozen tj mozemo da ponovo tokom treninga menjamo tezine grana
#ili ne.

if __name__ == "__main__":
    create_model()



#IZ DELA 6
#Postoji tesnorflow_datasets odakle mogu da se povuku neki test i training podaci
#Neke prevelike datasetove treba da podelimu u pakete - batch da bi ubrzali proces obrade
#Svako preuzimanje zahteva cpu i ram koji bi bili pretrpani ovako podelimo u odredjeni broj paketa iste velicine
#Podatke mozemo podeliti u paketa takodje ih spremiti i odraditi prefetch tj kada izdelimo podatke neki ce se obradjivati
#pa ce se cekati na cpu dok se obradjuje jedan paket priprema se drugi itd. Postoji mogucnost da se stavi u cache da se jos ubrza
#ali sve zavisi od resursa koje posedujemo
#Kreiranje callback funkcija nam omogucava da stanemo za evaluacijom modela ako je ptorebo par dana da  se trenira model
#takodje mozemo da stujemo ako na u narednih par epoha ne konvergira da presecemo jer dolazimo u zasicenje
#Mixed precision je kada korisitmo float32 ali da bi ubrzali stvari castujemo u float16 i kasnije reprezentujemo ponovo u 32
