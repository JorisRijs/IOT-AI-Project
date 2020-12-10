import os
import PIL
import PIL.Image
import cv2
import time
import tensorflow as tf
import pathlib
import numpy as np
from tensorflow.keras import layers
from keras_preprocessing import image



DATA_DIR = os.path.join(os.path.curdir, "Output")
data_dir = pathlib.Path(DATA_DIR)
checkpoint_path = 'training_1/cp.ckpt'
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True  # to log device placement (on which device the operation ran) 
sess = tf.compat.v1.Session(config=config)  
tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras 

batch_size = 2
img_height = 240
img_width = 320 

# Callback that saves the models weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)


training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1./255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1./255)
train_generator = training_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical"
)
validation_generator = training_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    class_mode="categorical"
)
def create_model(img_height, img_width):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 Neuron hidden Layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(11, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model

model = create_model(img_height, img_width)

exists = os.path.exists(os.path.join(os.path.curdir,'recognition.h5'))
if not exists:
    print("a saved model does not exists, we will need to train a new model and save it")
    model.fit(x=train_generator, epochs=25,
                                batch_size=batch_size,
                                validation_data= validation_generator,
                                callbacks=[cp_callback])
    os.system('mkdir -p saved_model')
    model.save('recognition.h5')
else:
    print("a trained model exists, so we will use that")
    model = tf.keras.models.load_model('recognition.h5')


model.summary()

def capture_frame():
    # cv2.CAP_DSHOW sets the capture device
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    print("printing frame properties")
    print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    check, frame = video.read()
    print(type(frame))
    video.release()
    return video

def feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    i = 0
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        #cv2.imshow('frame', gray)
        cv2.imshow('Webcam', frame)

        #print(frame)

        #geef aan per hoeveel frames het model orientatie moet checken
        if i == 5:
            print('frame')
            np_image_data = np.asarray(frame)
            x = np.expand_dims(frame, axis=0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
            print(classes)
            i = 0
            #print(AImodelresult(frame)) #links-11 rechts-11 etc.

        #framecapture()
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

feed()

# counter = 0
# running = True
# stop = False
# cap =cv2.VideoCapture(0)
# while running and not stop:
#     # Get a frame from the webcam
#     frame = capture_frame()
#     # Display the resulting frame
#     cv2.imshow('Webcam', frame)

#     if counter == 4:
#         print(frame)
#         np_image_data = np.asarray(frame)
#         x = np.expand_dims(frame, axis=0)
#         images = np.vstack([x])
#         classes = model.predict(images, batch_size=2)
#         counter = 0
#     else:
#         continue

#     counter += 1

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # When everything done, release the capture
# cv2.destroyAllWindows()

# test_path = os.path.join(os.path.curdir, 'test_data')

# for filename in os.listdir(test_path):
#     if filename.endswith('.jpg'):
#         file = os.path.join(test_path, filename)
#         print(os.path.join(test_path, filename))
#         img = image.load_img(file, target_size=(img_height, img_width))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         images = np.vstack([x])
#         classes = model.predict(images, batch_size=2)
#         print(classes)
#     else:
#         continue