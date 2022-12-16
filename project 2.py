import os.path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import socket
import time
import re
import cv2


HOST_IP = "192.168.1.103"
port = input("Enter port:")
if len(port) == 4:
    print("Valid port number")
while len(port) > 4 or len(port) < 4:
    print("Invalid port number")
    port = (input("Enter correct port: "))

pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST_IP, int(port)))

server.listen(5)

print(f"Your ip is {HOST_IP} and your port is {port}")
time.sleep(1)

print(f"welcome to Phora")
t = int(input("Epoch number: "))
model = tf.keras.models.load_model('Phora')
mn = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mn.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=t)
loss, accuracy = model.evaluate(x_test, y_test)

model.save('PHORA')

print(f"Your loss is {loss}")
print(f"Your accuracy is {accuracy}")
image_number = 1
while os.path.isfile('digits/digit.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1
    finally:
        print(f"here is your results {prediction}")
