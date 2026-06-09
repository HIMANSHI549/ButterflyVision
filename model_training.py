import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (
        (2.0 * intersection + smooth)
        /
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    )


def mean_iou(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    union = (
        K.sum(y_true_f)
        + K.sum(y_pred_f)
        - intersection
    )

    return (
        (intersection + smooth)
        /
        (union + smooth)
    )
class UNetTrainer:

    def __init__(
            self,
            input_shape=(256, 256, 3),
            num_filters=16,
            dropout_rate=0.07):

        self.input_shape = input_shape
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

    def convolutional_block(
            self,
            input_tensor,
            num_filters):

        x = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal"
        )(input_tensor)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(
            num_filters,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal"
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        return x
    

    
    def build_model(self):

        inputs = tf.keras.layers.Input(self.input_shape)

        # Encoder
        c1 = self.convolutional_block(inputs, 16)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
        p1 = tf.keras.layers.Dropout(self.dropout_rate)(p1)

        c2 = self.convolutional_block(p1, 32)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
        p2 = tf.keras.layers.Dropout(self.dropout_rate)(p2)

        c3 = self.convolutional_block(p2, 64)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
        
        p3 = tf.keras.layers.Dropout(self.dropout_rate)(p3)
        

        c4 = self.convolutional_block(p3, 128)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
        p4 = tf.keras.layers.Dropout(self.dropout_rate)(p4)

        # Bottleneck
        c5 = self.convolutional_block(p4, 256)

        # Decoder
        u6 = tf.keras.layers.Conv2DTranspose(
            128,
            (3, 3),
            strides=(2, 2),
            padding="same"
        )(c5)

        u6 = tf.keras.layers.concatenate([u6, c4])
        u6 = tf.keras.layers.Dropout(self.dropout_rate)(u6)

        c6 = self.convolutional_block(u6, 128)

        u7 = tf.keras.layers.Conv2DTranspose(
            64,
            (3, 3),
            strides=(2, 2),
            padding="same"
        )(c6)

        u7 = tf.keras.layers.concatenate([u7, c3])
        u7 = tf.keras.layers.Dropout(self.dropout_rate)(u7)

        c7 = self.convolutional_block(u7, 64)

        u8 = tf.keras.layers.Conv2DTranspose(
            32,
            (3, 3),
            strides=(2, 2),
            padding="same"
        )(c7)

        u8 = tf.keras.layers.concatenate([u8, c2])
        u8 = tf.keras.layers.Dropout(self.dropout_rate)(u8)

        c8 = self.convolutional_block(u8, 32)

        u9 = tf.keras.layers.Conv2DTranspose(
            16,
            (3, 3),
            strides=(2, 2),
            padding="same"
        )(c8)

        u9 = tf.keras.layers.concatenate([u9, c1])
        u9 = tf.keras.layers.Dropout(self.dropout_rate)(u9)

        c9 = self.convolutional_block(u9, 16)

        outputs = tf.keras.layers.Conv2D(
            1,
            (1, 1),
            activation="sigmoid"
        )(c9)

        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                dice_coefficient,
                mean_iou
            ]
        )

        return model

    def train(
            self,
            model,
            X_train,
            y_train,
            X_val,
            y_val):

        callbacks = [

            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=5
            ),

            ModelCheckpoint(
                "models/best_unet.keras",
                save_best_only=True
            )
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            callbacks=callbacks
        )

        return history