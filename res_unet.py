#from tf.keras.models import Model
#from tf.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, add, concatenate
import tensorflow as tf

def res_block(x, nb_filters, strides):
    res_path = tf.keras.layers.BatchNormalization()(x)
    res_path = tf.keras.layers.Activation(activation='relu')(res_path)
    res_path = tf.keras.layers.Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = tf.keras.layers.BatchNormalization()(res_path)
    res_path = tf.keras.layers.Activation(activation='relu')(res_path)
    res_path = tf.keras.layers.Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = tf.keras.layers.Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    res_path = tf.keras.layers.add([shortcut, res_path])
    return res_path


def encoder(x, filters):
    to_decoder = []

    main_path = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = tf.keras.layers.BatchNormalization()(main_path)
    main_path = tf.keras.layers.Activation(activation='relu')(main_path)

    main_path = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    main_path = tf.keras.layers.add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [filters*2, filters*2], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [filters*4, filters*4], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder, filters):
    main_path = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    main_path = tf.keras.layers.concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [filters*4, filters*4], [(1, 1), (1, 1)])

    main_path = tf.keras.layers.UpSampling2D(size=(2, 2))(main_path)
    main_path = tf.keras.layers.concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [filters*2, filters*2], [(1, 1), (1, 1)])

    main_path = tf.keras.layers.UpSampling2D(size=(2, 2))(main_path)
    main_path = tf.keras.layers.concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [filters, filters], [(1, 1), (1, 1)])

    return main_path


def build_res_unet(input_shape, masks_shape, filters=64, channel =None):
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    masks = tf.keras.layers.Input(shape=masks_shape)
    
    to_decoder = encoder(inputs, filters)

    path = res_block(to_decoder[2], [filters*8, filters*8], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder, filters=filters)
    
    

    #path = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1),activation='softmax')(path)
    
    path = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1))(path)
    
    
    path = tf.keras.layers.Multiply()([path,masks])
    
    path = tf.keras.layers.Activation(activation='softmax')(path)

    return tf.keras.models.Model(inputs=[inputs,masks], outputs=path)
