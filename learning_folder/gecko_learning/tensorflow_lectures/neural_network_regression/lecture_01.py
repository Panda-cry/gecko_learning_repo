import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# NN neural network sastoji se od par stvari. Da bi ona sama postojala potrebni su nam ulazni parametri i potrebno je da znamo kakav ce izlaz biti
# Za samu NN postoje slojevi, prvi sloj je input sloj gde upadaju  podaci nakon toga nalaze se hidden layers slojevi koje ne vidimo ali sluze za bolje racunanje paterna koji treba da nauci NN
# Broj slojeva zavisi takodje kao i broj neurona po slojevima i a kraju sloj output tj izlazni sta je dobitak
# NN ima tri kao pdovrste kada imamo pod nazorom NN sto znaci da imamo vec neki output pa nase predikcije mozemo da uporedimo,
# semi pola pola i kada nemamo uposte znanje kakav izlaz ce biti. Za rad koristi se kerasov Sequence a slojevi su Dence
# svaki sloj ima aktivacionu fju sa kojoj se racuna ima ih dosta Relu,Mean squered error, Abslout mean squared error itd.
# tokom kompajliranja modela funcija loss biramo kojom funkcijom cemo videti koliko gresimo otprilike a sa optimizer geldamo kojom fjom to da popravimo i kako.
# metrics samo nam govori tj ispisuje te gubitke razliku prediktovane vrednosti od stvarne
# kod fitovanja modela tj pustanja da uci model epohe su koliko se vrtimo tj koliko model uci
# Osnovni neki zadaci kada se pogresi je dodati vise/manje slojeva, promeniti funkciju optimizacije ,promeniti malo data, ucenje duze/manje,
# ucenje nad odredjenim setom, promena aktivacionih fja

# X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
#
# # Create labels
# y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it
# plt.scatter(X, y)
# plt.show()

# tf.random.set_seed(42)
# Kreiramo prvi model
# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Dense(1),
#     ]
# )
#
# model.compile(loss=tf.keras.losses.mae,
#               optimizer=tf.keras.optimizers.SGD(),
#               metrics=["mae"])
#
# model.fit(tf.expand_dims(X, axis=-1),y,epochs=100)
#
# print(model.predict([17]))

# Kreiracemo model samo sa vecim datasetom
# #
# X = np.arange(-100, 100, 4)
# y = np.arange(-90, 110, 4)
# tf.random.set_seed(42)
# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Dense(1),
#     ]
# )
#
# model.compile(loss=tf.keras.losses.mae,
#               optimizer=tf.keras.optimizers.SGD(),
#               metrics=["mae"])
#
# model.fit(tf.expand_dims(X,axis=-1),y,epochs=100)
# print(model.predict([17.0]))
# Potrebana je podela podataka na testne podatke i na trening podatka od 75% do 90 posto treba da bude trening podataka
# Podela podataka
# x_train = X[:40]
# x_test = X[40:]
# y_train = y[:40]
# y_test = y[40:]
#
#
# plt.scatter(x_train, y_train, c='b', label='Training data')
# # Plot test data in green
# plt.scatter(x_test, y_test, c='g', label='Testing data')
# # Show the legend
# plt.legend()
# # plt.show()
#
#tf.random.set_seed(42)
# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Dense(1),
#     ]
# )
#
# model.compile(loss=tf.keras.losses.mae,
#               optimizer=tf.keras.optimizers.SGD(),
#               metrics=["mae"])
#
# model.fit(tf.expand_dims(x_train, axis=-1), y_train, epochs=100,verbose=0)
# # model.summary()
# print(model.predict([17.0]))

# Mozemo da napravimo malu skicu ulaza i izlaza naseg
# tf.keras.utils.plot_model(model, show_shapes=True)
# Nakon svega mozemo da postavimo predikcije tako da ubacimo listu i iz nje prediktujemo nove izlaze
# y_predicted = model.predict(x_test)

# Napracicemo fju za plotovanje svega
# from typing import List, Any
#
#
# def _plot_results(x_train: List[Any] = x_train,
#                   x_test: List[Any] = x_test,
#                   y_train: List[Any] = y_train,
#                   y_test: List[Any] = y_test,
#                   y_predicted: List[Any] = None):
#     plt.figure(figsize=(10, 7))
#     plt.scatter(x_train, y_train, c="b", label="Training data")
#     plt.scatter(x_test, y_test, c="y", label="Test data")
#     plt.scatter(x_test, y_predicted, c="g", label="Predicted data")
#     plt.legend()
#     plt.show()


# _plot_results(x_train,x_test,y_train,y_test,y_predicted)
# Nakon svake obrade modela izvodi se procena evaluacija modela
# Dva glavna nacina procene modela su MAE MSE Mean absloute error i mean squared error
# model.evaluate(y_test,y_predicted)

# Posto smo koristili mae i kao loss funkciju i za metrics za evaluaciju mozemo da izracunamo
# predikciju sa verednostima koje bi trebalo da budu
# mae = tf.metrics.mean_absolute_error(y_true=y_test,y_pred=y_predicted)
# print(mae)
# print("-------------")
# print(y_test.shape)
# print(y_predicted.shape)
# print("-------------")
# y_predicted = tf.squeeze(y_predicted)
# mae = tf.metrics.mean_absolute_error(y_true=y_test,y_pred=y_predicted)
# print(mae)
# Npr ovde mae treba da bude samo jedan broj
# Posto imamo problem sa ne komaptibilnim listama y_predict je dim 1 dok je y_test dim 0 a mean treba da sabira sve i onda da deli
# onda ce ona napraviti 10 mean-ova jer je jedna dim vise pa ce ovaj uzimati sve test i porediti sa jednom po jednom vrednoscu
# i tako dobijamo opet 10 elemenata a mae treba da bude 1 vrednost :D, uvek paziti na dimenzije tj na input i output
# Sada cemo porediti 3 modela :D
# def _mae(y_prediceted):
#     return tf.metrics.mean_absolute_error(y_test, tf.squeeze(y_prediceted))


#
# model_1 = tf.keras.Sequential(
#     [
#         tf.keras.layers.Dense(1)
#     ]
# )
#
# model_1.compile(
#     loss=tf.keras.losses.mae,
#     optimizer=tf.keras.optimizers.SGD(),
#     metrics=['mae']
# )
#
# model_1.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100,verbose=0) #Verbose 0 je da nam ne ispisuje sve ono kako radi
#
# y_predicted_model_1 = model_1.predict(x_test)
# _plot_results(y_predicted=y_predicted_model_1)
# print(f"Mae for model 1 is :{_mae(y_predicted_model_1)}")
# #print(f"Evaluation for model 1 is {model_1.evaluate(x_test,y_test)}")#Evaluate pokazuje i loss i metrics
# print("----------------------")
# print()
# model_2 = tf.keras.Sequential(
#     [
#         tf.keras.layers.Dense(1),
#         tf.keras.layers.Dense(1)
#     ]
# )
#
# model_2.compile(
#     loss=tf.keras.losses.mae,
#     optimizer=tf.keras.optimizers.SGD(),
#     metrics=['mae']
# )
#
# model_2.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=100,verbose=0) #Verbose 0 je da nam ne ispisuje sve ono kako radi
#
# y_predicted_model_2 = model_2.predict(x_test)
# _plot_results(y_predicted=y_predicted_model_2)
# print(f"Mae for model 2 is :{_mae(y_predicted_model_2)}")
# print('---------------------')
# print()
#
# model_3 = tf.keras.Sequential(
#     [
#         tf.keras.layers.Dense(1),
#         tf.keras.layers.Dense(1)
#     ]
# )
#
# model_3.compile(
#     loss=tf.keras.losses.mae,
#     optimizer=tf.keras.optimizers.SGD(),
#     metrics=['mae']
# )
#
# model_3.fit(tf.expand_dims(x_train,axis=-1),y_train,epochs=500,verbose=0) #Verbose 0 je da nam ne ispisuje sve ono kako radi
#
# y_predicted_model_3 = model_3.predict(x_test)
# _plot_results(y_predicted=y_predicted_model_3)
# print(f"Mae for model 2 is :{_mae(y_predicted_model_3)}")

# Cuvanje modela sa .save() default foramt je bolji jer cuva sve trenutne layers i ne dodaje nista kada je u .h5 formatu mora nesto malo da se menja
# load isti nacin dobije se model i tjt :D


# Sada ide veci primer i vezbanje modela
import pandas as pd
import numpy as np

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# Posto imamo sex, region i jos jedan field da je string based trebamo pretvoriti to u numricko 0,1 da bi NN mogla da uci pravilno
# u pandasu postoji metoda get_dummies() koja ne numericka polja pretvara u hot_encoded nule i jedinice
#
# insurance_numeric = pd.get_dummies(insurance)
# insurance_numeric = insurance_numeric.astype(np.float32)
# # potrebna je podela na X i na y zavisne i nezaisne promenljive
# # posto treba da naucimo mrezu da proracuna koliko kosta y ce biti suma a sve ostalo X
# #Problem sada sa ovim tenserflow ili sa pandas jer ne prepoznaje bool polje koje je to int buni se :D
# X = insurance_numeric.drop("charges",axis=1)
# y = insurance_numeric["charges"]
from sklearn.model_selection import train_test_split

#
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

# Prvi test tf.keras.layers.Dense(1),
# tf.keras.layers.Dense(1),
insurance_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(200),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(50),
        tf.keras.layers.Dense(1),
    ]
)
# #Prvi test SGD()
insurance_model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["mae"]
)
# history = insurance_model.fit(X_train, Y_train, epochs=100,verbose=0)
# insurance_model.evaluate(X_test,Y_test)
#
# pd.DataFrame(history.history).plot()
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.show()
#
# history2 = insurance_model.fit(X_train, Y_train, epochs=100,verbose=0)
# pd.DataFrame(history2.history).plot()
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.show()

# Trebalo bi da se skaliraju sve vrednosti moze normalizacija vrednosti tj da budu izmedju 0,1 ili standardizacija
# da budu oko srednje vrednosti
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

X = insurance.drop("charges", axis=1)
y = insurance["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# Ovo se radi samo na train podacima na test moze da dodje do gubitaka podataka
ct.fit(X_train)

X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

history_3 = insurance_model.fit(X_train_normal, y_train, epochs=300, verbose=0)

pd.DataFrame(history_3.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

print(insurance_model.evaluate(X_test_normal, y_test))
# Benefit skaliranja tj normalizacije je brza konvergencija ka resenju tj ubrzanje za oko 10%
# Novo sto sam naucio je da postoji fja koja moze da zameni non numerical polja sa one_hot_encoded poljima
# make_column_transformer moze da skalira i da enkoduje tj prosiri odmah
