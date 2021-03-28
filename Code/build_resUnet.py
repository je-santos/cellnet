
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D,  \
            BatchNormalization, Activation, Add, Multiply, \
            concatenate, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape

'''
Contains all the necessary components to build Cellnet, A hybrid ResUnet
'''


def build_cellnet(input_shape, masks_shape, nuclei_shape, filters=64,
                  SnE=False, incl_attn=False, incl_pvp=False,
                  incl_masks=True, num_heads=1):

    inputs = Input(shape=input_shape)
    masks = Input(shape=masks_shape)
    nuclei = Input(shape=nuclei_shape)

    to_decoder = encoder(inputs, filters, SnE=SnE)

    path = agg_decoder(masks, to_decoder, filters, incl_pvp, incl_attn,
                       incl_masks, name='cells')

    if num_heads == 2:
        path_nuclei = agg_decoder(nuclei, to_decoder, filters, incl_pvp,
                                  incl_attn, incl_masks, name='nuclei')
        return tf.keras.models.Model(inputs=[inputs, masks, nuclei],
                                     outputs=[path, path_nuclei])

    return tf.keras.models.Model(inputs=[inputs, masks], outputs=path)


def agg_decoder(masks, to_decoder, filters, incl_pvp, incl_attn,
                incl_masks, name):
    dec_path = res_block(to_decoder[2], [filters*8, filters*8],
                         [(2, 2), (1, 1)])
    if incl_pvp is True:
        dec_path = pyramidal_pool(dec_path, filters*8)

    dec_path = decoder(dec_path, from_encoder=to_decoder, filters=filters,
                       incl_attn=incl_attn)

    dec_path = Conv2D(filters=2, kernel_size=(1, 1))(dec_path)

    if incl_masks is True:
        dec_path = Multiply()([dec_path, masks])

    dec_path = Activation(activation='softmax', name=f'{name}')(dec_path)
    return dec_path


def pyramidal_pool(x, filters):
    """
    Contextual info captured at multiple scales and then fused

    LAST_ENC -> ASPP -> FIRST_DEC

    Note:
    The current dilation rate could be multiplied by a factor
    """

    p1 = Conv2D(filters, (3, 3), dilation_rate=(1, 1), padding='same',
                use_bias=False)(x)
    p1 = BatchNormalization()(p1)

    p2 = Conv2D(filters, (3, 3), dilation_rate=(6*1, 6*1), padding='same',
                use_bias=False)(x)
    p2 = BatchNormalization()(p2)

    p3 = Conv2D(filters, (3, 3), dilation_rate=(6*2, 6*2), padding='same',
                use_bias=False)(x)
    p3 = BatchNormalization()(p3)

    p4 = Conv2D(filters, (3, 3), dilation_rate=(6*3, 6*3), padding='same',
                use_bias=False)(x)
    p4 = BatchNormalization()(p4)

    p = Add()([p1, p2, p3, p4])
    p = Conv2D(filters, (1, 1), padding='same')(p)
    return p


def attn(enc, dec):
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
    enc_conv = MaxPooling2D(pool_size=(2, 2), strides=(1, 1),
                            padding='same')(enc_conv)

    branch_sum = Add()([dec_conv, enc_conv])
    branch_sum = BatchNormalization()(branch_sum)
    branch_sum = Activation("relu")(branch_sum)
    branch_sum = Conv2D(filters, (3, 3), padding='same')(branch_sum)

    activated = Multiply()([dec, branch_sum])
    return activated


def S_n_E(input_tensor):
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
           |
          S&E
           |
           |
           |
          ENC

    """
    init = input_tensor
    filters = init.shape[-1]

    sne = GlobalAveragePooling2D()(init)
    sne = Reshape((1, 1, filters))(sne)
    sne = Dense(filters//8, activation='relu', kernel_initializer='he_normal',
                use_bias=False)(sne)
    sne = Dense(filters, activation='sigmoid', kernel_initializer='he_normal',
                use_bias=False)(sne)

    x = Multiply()([init, sne])
    return x


def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3),
                      padding='same', strides=strides[0],
                      use_bias=False)(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3),
                      padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0],
                      use_bias=False)(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = tf.keras.layers.add([shortcut, res_path])
    return res_path


def encoder(x, filters, SnE):
    to_decoder = []

    main_path = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
                       strides=(1, 1), use_bias=False)(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
                       strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1),
                      use_bias=False)(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = tf.keras.layers.add([shortcut, main_path])
    # first branching to decoder
    to_decoder.append(main_path)
    if SnE is True:
        main_path = S_n_E(main_path)

    main_path = res_block(main_path, [filters*2, filters*2], [(2, 2), (1, 1)])
    to_decoder.append(main_path)
    if SnE is True:
        main_path = S_n_E(main_path)

    main_path = res_block(main_path, [filters*4, filters*4], [(2, 2), (1, 1)])
    to_decoder.append(main_path)
    if SnE is True:
        main_path = S_n_E(main_path)

    return to_decoder


def decoder(x, from_encoder, filters, incl_attn):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [filters*4, filters*4], [(1, 1), (1, 1)])

    if incl_attn is True:
        main_path = attn(main_path, from_encoder[2])
    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [filters*2, filters*2], [(1, 1), (1, 1)])

    if incl_attn is True:
        main_path = attn(main_path, from_encoder[1])
    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [filters, filters], [(1, 1), (1, 1)])

    return main_path
