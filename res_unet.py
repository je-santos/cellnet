#from tf.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D,  \
            BatchNormalization, Activation, Add, Multiply, \
            concatenate, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape



"""
This file contains the necessary components to create cellnet:
A hybrid between a REsUnet and a ResUnet++
...
"""



def pyramidal_pool(x,filters):
    """
    Contextual info captured at multiple scales and then fused

    LAST_ENC -> ASPP -> FIRST_DEC
    
    
    Note:
    The current dilation rate could be multiplied by a factor
    """
    
    p1 = Conv2D(filters, (3, 3), dilation_rate=(1,1), padding='same')(x)
    p1 = BatchNormalization()(p1)
    
    p2 = Conv2D(filters, (3, 3), dilation_rate=(6*1,6*1), padding='same')(x)
    p2 = BatchNormalization()(p2)

    p3 = Conv2D(filters, (3, 3), dilation_rate=(6*2,6*2), padding='same')(x)
    p3 = BatchNormalization()(p3)
    
    p4 = Conv2D(filters, (3, 3), dilation_rate=(6*3,6*3), padding='same')(x)
    p4 = BatchNormalization()(p4)
    
    p = Add()([p1, p2, p3, p4])
    p = Conv2D(filters, (1, 1), padding='same')(p)
    return p
   

def attn(enc,dec):
    """
    It highlights a portion of the image. It's used in the decoder as follows:


        ENC  ------>  Attention_layer
                            ^
                            |
                            |
                           DEC

    """
    
    filters = dec.shape[-1]
    
    dec_conv = BatchNormalization()(dec)
    dec_conv = Activation("relu")(dec_conv)
    dec_conv = Conv2D(filters, (3, 3), padding='same')(dec_conv)
    
    enc_conv = BatchNormalization()(enc)
    enc_conv = Activation("relu")(enc_conv)
    enc_conv = Conv2D(filters, (3, 3), padding='same')(enc_conv)
    enc_conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(enc_conv)
    
    branch_sum = Add()([dec_conv, enc_conv])
    branch_sum = BatchNormalization()(branch_sum)
    branch_sum = Activation("relu")(branch_sum)
    branch_sum = Conv2D(filters, (3, 3), padding='same')(branch_sum)
    
    activated = Multiply()([dec, branch_sum])
    return activated


def S_n_E(x):
    """
    ----------------------
    Squeeze and Excitation
    ----------------------
    
    Models interdependencies between channels to recalibrate their responses. 
    This is achieved by using global average pooling to get per channel stats 
    and then the channel response is recalibrated.
    
          ENC
           |------->
           |-------> Attn
          ▽
          S&E
           |
           |
          ▽
          ENC

    """
    filters = x.shape[-1]
    
    sne = GlobalAveragePooling2D()(x)
    sne = Reshape((1,1,filters))(sne)
    sne = Dense(filters//8, activation='relu', kernel_initializer='he_normal', use_bias=False)(sne)
    sne = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(sne)

    x = Multiply()([x, sne])


def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = Add([shortcut, res_path])
    return res_path


def encoder(x, filters):
    to_decoder = []

    main_path = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = Add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)

    main_path = res_block(main_path, [filters*2, filters*2], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block(main_path, [filters*4, filters*4], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder


def decoder(x, from_encoder, filters):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [filters*4, filters*4], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [filters*2, filters*2], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [filters, filters], [(1, 1), (1, 1)])

    return main_path


def stem_block(x, n_filter):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same")(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same")(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = S_n_E(x)
    return x





def build_cellnet(input_shape, filters=64):
    inputs = Input(shape=input_shape)
    """
    To be continued
    """




def build_res_unet(input_shape, masks_shape, filters=64):
    
    inputs = Input(shape=input_shape)
    
    masks = Input(shape=masks_shape)
    
    to_decoder = encoder(inputs, filters)

    path = res_block(to_decoder[2], [filters*8, filters*8], [(2, 2), (1, 1)])

    path = decoder(path, from_encoder=to_decoder, filters=filters)
    
    
    path = Conv2D(filters=2, kernel_size=(1, 1))(path)
    
    
    path = Multiply()([path,masks])
    
    path = Activation(activation='softmax')(path)

    return tf.keras.models.Model(inputs=[inputs,masks], outputs=path)



def build_2_head_unet(input_shape, masks_shape, nuclei_shape, filters=64, channel =None):
        
    inputs = Input(shape=input_shape)
    
    masks = Input(shape=masks_shape)
    
    nuclei = Input(shape=nuclei_shape)
    
    to_decoder = encoder(inputs, filters)

    path_cell = res_block(to_decoder[2], [filters*8, filters*8], [(2, 2), (1, 1)])
    path_nuc  = res_block(to_decoder[2], [filters*8, filters*8], [(2, 2), (1, 1)])
    
    path_cell = decoder(path_cell, from_encoder=to_decoder, filters=filters)
    path_nuc  = decoder(path_nuc,  from_encoder=to_decoder, filters=filters)
    
    path_cell = Conv2D(filters=2, kernel_size=(1, 1))(path_cell)
    path_nuc  = Conv2D(filters=2, kernel_size=(1, 1))(path_nuc)
    
    path_cell = Multiply()([path_cell,masks])
    path_nuc  = Multiply()([path_nuc, masks])
    
    path_cell = Activation(activation='softmax')(path_cell)
    path_nuc  = Activation(activation='softmax')(path_nuc)
    
    return tf.keras.models.Model(inputs=[inputs,masks,nuclei], outputs=[path_cell, path_nuc])
    
