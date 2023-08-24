import tensorflow as tf

# Primeri su sa: https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/00_tensorflow_fundamentals.ipynb

# Intro sa Tensorflow
# ML ili machine learning  predstavlja jedan deo AI mozemo zamisliti to preko skupova
# Najveci skup je AI manji skup u njemu je ML dok je DL deep learning najmanji skup svega toga.
# Razlika izmedju ML i DL je da je DL zasniva na neruonskoj mrezi i algoritmima vezanih za DL.
# ML poseduje veci broj algoritama kao sto su najblizi susedi, algoritam sume itd.
# Tensorflow se koristi jer je lako preci na GPU radi brzeg izracunavanja.
# DL se moze predstaviti sve na ovom svetu samo ako mozemo da zapisemo nase razmisljanje u 0,1
# DL, ML je ustvari pronalazenje slicnosti u podacima i tako stvaranje patterna preko kojeg mozemo da izracunamo dalje kombinacije.
# Napomena ako hocemo da samnjimo tacnost, a povecamo performanse promena sa int64 ili int32 na manji int npr in16 ili bint16.

# Trenutna verzija TensorFlow
print(tf.__version__)

# Svakako necemo mi tokom DL kreirati tensore vec ce to automatski da se radi ali radi reda i upoznavanja sa tematikom malo radimo sa tensorima
# tensor je neki pdoatak matrica, vektor skup brokjeva koji opisuju nesto
# Create tensor
scalar = tf.constant(7)
print(scalar)
print(scalar.ndim)
# ovo je konstanta nema neku dimezniju
# takodje constant ne moze da se menaj tokom rada

vector = tf.constant([1, 2, 3])
print(vector)
print(vector.ndim)
# vektor je obicna lista i dimenzija mu je 1
# ovde npr vector.shape daje tuple(3,) gde 3 predstavlja broj elemenata

matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])
print(matrix)
print(matrix.ndim)
# matrix je dimenzija 2 dok shape je 2,3 shape se redja 2 jer imamo 2 liste da imamo vise dimenzija bili bi broj dimenzija,
# broj elemenata dimenzije ispod itd do broj elemenata

# po defaultu je  int32 ili float32 sto je veci broj zauzima vise memorije i preciziniji je
# mozemo da definisemo kojeg ce tipa biti
another_matrix = tf.constant([[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.]], dtype=tf.float16)
print(another_matrix)
print(another_matrix.ndim)

# Na primer ovo je shape(3,2,3) jer imamo 3 liste kao unutar, pa u svakoj listi imamo po jos 2 lite i na kraju u svakoj donjoj listi po 3 elementa
# broj dimenzija gledamo po ranku iz algebre ili po shape koliko ima elemenata :D ova ima 3 dimenzije
tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [10, 11, 12]],
                      [[13, 14, 15],
                       [16, 17, 18]]])
print(tensor)

# skalar je broj, vektor je broj sa brzinom i smerom, matrica je 2 dimenzionalni red, a tensor je n dimenzionalni red
# tensore mozemo kreirati i sa tf.Variable razlika od constant jer je constant imutabilan dok je ovaj mutablilan

# changeble tenstor
changable_tensor = tf.Variable([10, 7])

changable_tensor[0].assign(7)
# promenili smo mesto 10 na 7
print(changable_tensor)

# kreiranje random tensora
# random tensori se krosiste kao inicijalizacija tezina grana izmedju neurona za pocetak

random_1 = tf.random.Generator.from_seed(7)
random_2 = tf.random.Generator.from_seed(7)
random_1 = random_1.normal(shape=(3, 3))
random_2 = random_2.normal(shape=(3, 3))

print("--------------\n")
print(random_1)
print()
print(random_2)
print(random_1 == random_2)

# kada bi stavili jos jedan globalni seed uvek bi isti brojevi izvbijali

# potrebno je da se suffle izmesa malo da bi bolje neuronska mreza radila

not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [2, 5]])
# Stalno drugacije dobijamo
shuffled = tf.random.shuffle(not_shuffled)
print(shuffled)

# stalno bi isto dibijli
shuffled = tf.random.shuffle(not_shuffled, seed=42)
print(shuffled)
# Kada bi globalni seed stavili ne bi se izmesalo
tf.random.set_seed(42)
shuffled = tf.random.shuffle(not_shuffled, seed=42)
print(shuffled)

# takodje kao i mozemo tensor da pravimo od matrice zeros i ones
# tensroflow lepo radi sa numpy

print(tf.zeros(shape=(3, 3)))
print(tf.ones(shape=(3, 3)))
# mozemo red tj numpy array da pretvorimo u shape koji hocemo
import numpy as np

np_array = np.arange(1, 25, dtype=np.int32)
tensor_A = tf.constant(np_array, shape=(2, 3, 4))

print(tensor_A)

rank_4_tensor = tf.zeros([2, 3, 4, 5])
print(rank_4_tensor)
# shape ide redom samo se spaja broj cega ima :D
# Get various attributes of tensor
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (2*3*4*5):", tf.size(rank_4_tensor).numpy())

print("Get 2 item from each diemnsion")
print(rank_4_tensor[:2, :2, :2, :2])

print("Get the dimension from each index except for the final one")
print(rank_4_tensor[:1, :1, :1, :])

# Create a rank 2 tensor (2 dimensions)
rank_2_tensor = tf.constant([[10, 7],
                             [3, 4]])

print(f"Poslednji elementi tensora 2 dimenzije su  : {rank_2_tensor[:, -1]}")

# axis govori gde cemo da dodamo tj izmedju koje dimeznije cemo umetnutu 0... -1
rank_3_tensor = rank_2_tensor[..., tf.newaxis]  # in Python "..." means "all dimensions prior to"
print(rank_3_tensor)
rank_3_tensor = tf.expand_dims(rank_2_tensor, axis=1)
print(rank_3_tensor)

# + * - / na sve elemente dodaje taj broj *10 mnozi sve sa 10 sve elemente

X = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])
# mnozenje matrica mora da unutrasnje budu iste npr ako je 3x2 mora da se mnozi sa 2x3
# transpose nije isto sto i reshape ja sam naucio da je prelepljivanje od dole na gore @ je za mnozenje matrica
# reshape samo ide 1,2,3 ,4,5,6, ona samo dopuni do ranga koji treba i naredni red
Y = tf.transpose(X)

print(X @ Y)

Y = tf.reshape(X, shape=(2, 3))
print(X @ Y)

print((X @ Y) == tf.matmul(X,Y))
#tf.mtamul moze da reshape a ili b tj one matrice sto prosledimo :D
print(tf.matmul(X,X,transpose_b=True))
#funkcija tf.tensordot isto se ponasa

#Kastovanje tipa tf.cast() promena za bolje performanse

B = tf.constant([1.7, 7.4])

C = tf.constant([1, 7])

print(tf.cast(B,dtype=tf.float16))
print(tf.cast(C,dtype=tf.int16))

#funkcije agregacije za odredjeni tensor
B1 = B * -1
print(B1)
print(tf.abs(B1))

#sve ostale fje kao sto su min max mean tj srednja vrednost suma tensora sve moze da se nadje na tf.math.reduce ili nesto naziv fje.
#takodje na kojoj se poziciji moze pronaci tf.argmax ili argmin
#samnjenje dimeznije tensora ako ima dosta 1 dimenzionalnih stvari tf.squeeze
#oneshoot ecnoding kada imamo neke stvari koje treba da reprezenujemo u bin
#neki string ili nesto tako da napravimo mozda listu i dubina je len(liste) pa ce biti kao brojanje binarno od 0 do max(liste)
#sve po brojevima kao tablica :D
#lagana konverzija iz tensora u numpy
#sve ovo nisam kucao jer je lagano :D


print(tf.config.list_physical_devices('GPU'))
