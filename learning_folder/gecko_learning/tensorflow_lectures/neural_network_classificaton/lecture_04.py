import datetime

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from keras import layers

class TransferLearningModel:

    BATCH_SIZE = 32
    INPUT_SHAPE = (224, 224)

    train_dir = "10_food_classes_10_percent/train/"
    test_dir = "10_food_classes_10_percent/test/"

    resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
    efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

    @classmethod
    def return_test_and_train_data(cls):
        train_generator = ImageDataGenerator(rescale=1/255.)
        test_generator = ImageDataGenerator(rescale=1/255.)

        train_data = train_generator.flow_from_directory(cls.train_dir,batch_size=cls.BATCH_SIZE,
                                                        target_size=cls.INPUT_SHAPE, class_mode="categorical")
        test_data = test_generator.flow_from_directory(cls.test_dir,batch_size=cls.BATCH_SIZE,
                                                        target_size=cls.INPUT_SHAPE, class_mode="categorical")

        return train_data, test_data


    @classmethod
    def create_model(cls, path, num_classes=10):
        m = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
                           trainable=False),  # Can be True, see below.
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        m.build([None, 224, 224, 3])

        return m

#Postoji mogucnost pronalaska modela za sta nam je potrebno tj vec istreniranog modela sa trensorflow.hub
#treba paziti samo da li su slicni podaci nad kojima se trenirao taj model i podaci koje mi imamo
#ostalo je sve manje vise isto 
    @classmethod
    def model_training(cls):
        train_data, test_data =  cls.return_test_and_train_data()
        resnet_model = cls.create_model(cls.efficientnet_url,num_classes=train_data.num_classes)

        resnet_model.compile(loss='categorical_crossentropy',
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=['accuracy'])

        resnet_history = resnet_model.fit(train_data,
                                          epochs=5,
                                          steps_per_epoch=len(train_data),
                                          validation_data=test_data,
                                          validation_steps=len(test_data),
                                          # Add TensorBoard callback to model (callbacks parameter takes a list)
                                          callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                                 # save experiment logs here
                                                                                 experiment_name="resnet50V2")])



def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


if __name__ == "__main__":

    TransferLearningModel.model_training()
