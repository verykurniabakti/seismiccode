from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from Library.metrics import wasserstein_1D, euclidean
import config

def conv_stride(inputs,
               kernel_num,
               kernel_size,
               stride_conv,
               pool_size,
               stride_pool,
               activation="relu"):

    _feature = keras.layers.Conv1D(kernel_num,
                                   kernel_size,
                                   strides=stride_conv,
                                   padding='same',
                                   activation=activation)(inputs)
    
    _merge_time = tf.reduce_mean(_feature, axis=-2, keepdims=True)

    _merge_depth = tf.reduce_mean(_feature, axis=-1, keepdims=True)
    _merge_depth = tf.transpose(_merge_depth, [0, 2, 1])

    _merged = tf.concat([_merge_time, _merge_depth], -1)

    avg_pool_1d = tf.keras.layers.MaxPool1D(pool_size=pool_size,
                                            strides=stride_pool,
                                            padding='valid',
                                            data_format='channels_first')
    
    output = avg_pool_1d(_merged)

    return output



def parse_genes(chromosome):
    
    # Partition of chromosome
    #----- resolution ----
    # (0) conv_stride, 
    #---- convolution ----
    # CAUTION: make sure at least one kernel gene > 0
    # (1) kernel_1,
    # (2) kernel_3,
    # (3) kernel_5,
    # (4) kernel_7,
    # (5) kernel_9,
    #--- feature compression ---
    # (6) pooling_size,
    # (7) pooling_stride,
    #--- multi layer perception ---
    # (8) layer_1,
    # (9) layer_2,
    # (10) layer_3,

    conv_params = list()
    pooling_params = list()
    mlp_params = list()


    # get convolution params
    conv_params.append(round(chromosome[0]))

    # kernel_1
    if chromosome[1]>0:
         conv_params.append(1)
    else:
         conv_params.append(0)

    # kernel_3
    if chromosome[2]>0:
         conv_params.append(1)
    else:
         conv_params.append(0)

    # kernel_5
    if chromosome[3]>0:
         conv_params.append(1)
    else:
         conv_params.append(0)

    # kernel_7
    if chromosome[4]>0:
         conv_params.append(1)
    else:
         conv_params.append(0)

    # kernel_9
    if chromosome[5]>0:
         conv_params.append(1)
    else:
         conv_params.append(0)

    # get pooling params
    pooling_params.extend([round(chromosome[6]),
                           round(chromosome[7]),
                           ])
    
    # get mlp params
    if round(chromosome[9]) <= 0:
         mlp_params.append(round(chromosome[8]))
    elif round(chromosome[10]) <= 0:
         mlp_params.extend([round(chromosome[8]), round(chromosome[9])])
    else:
         mlp_params.extend([round(chromosome[8]),
                            round(chromosome[9]),
                            round(chromosome[10]),
                            ])

    return conv_params, pooling_params, mlp_params


def build_features(inputs,
                input_size,   
                conv_params,
                pooling_params):
    
    # --------------------------------
    # Estimate valid number of kernels
    # --------------------------------
    spacial_resolution = round(input_size/conv_params[0])
    _temp_tensor = tf.random.normal((1, input_size, 1))
    _temp_feature =tf.keras.layers.Conv1D(filters = spacial_resolution,
                                        kernel_size = 3,
                                        strides=conv_params[0],
                                        padding='same')(_temp_tensor)
    _time_size = int(_temp_feature.shape[-2])
    
    if _time_size != spacial_resolution:
        spacial_resolution = _time_size

    # ------------------
    # Build feature maps
    # ------------------
    # kernel size list
    kernel_sizes = [1, 3, 5, 7, 9]

    # convolution ops
    feature_list = []
    for i in range(len(conv_params)-1):
        kernel_flag = conv_params[i+1]
        if kernel_flag == 1:
            _feature = conv_stride(inputs = inputs,
                                    kernel_num = spacial_resolution,
                                    kernel_size = kernel_sizes[i],
                                    stride_conv = conv_params[0],
                                    pool_size = pooling_params[0],
                                    stride_pool = pooling_params[1],
                                    activation=config.ACTIVITION)
            feature_list.append(_feature)

    concat_features = tf.concat(feature_list, 1)

    merged_features = keras.layers.GlobalMaxPooling1D()(concat_features)

    return merged_features



def transform_embedding(features, params):
    if len(params)==1:
        codes = keras.layers.Dense(units=params[0])(features)
    elif len(params)==2:
        x = keras.layers.Dense(units=params[0], activation=config.ACTIVITION)(features)
        x = keras.layers.Dropout(0.2)(x)
        codes = keras.layers.Dense(units=params[1])(x)
    else:
        x = keras.layers.Dense(units=params[0], activation=config.ACTIVITION)(features)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(units=params[1], activation=config.ACTIVITION)(x)
        x = keras.layers.Dropout(0.2)(x)
        codes = keras.layers.Dense(units=params[2])(x)

    return codes
         

def Embedding_Network(input_size,
                      chromosome,
                      ):
    
    conv_params, pooling_params, mlp_params = parse_genes(chromosome)

    inputs = keras.Input(shape=(input_size, 1))

    merged_features = build_features(inputs = inputs,
                                    input_size = input_size,
                                    conv_params=conv_params,
                                    pooling_params=pooling_params,
                                    )
    codes = transform_embedding(merged_features, mlp_params)

    embeddings = keras.Model(inputs, codes, name="embeddings")

    return embeddings


def Contrastive_Network(input_size, embedding_model):
    # input layers
    input_refer = keras.Input(name="refer", shape=(input_size, 1))
    input_pos = keras.Input(name="positive", shape=(input_size, 1))
    input_neg = keras.Input(name="negative", shape=(input_size, 1))
    input_sil = keras.Input(name="silence", shape=(input_size, 1))

    # get embeddings
    embedding_refer = embedding_model(input_refer)
    embedding_pos = embedding_model(input_pos)
    embedding_neg = embedding_model(input_neg)
    embedding_sil = embedding_model(input_sil)

    # build the network
    _network = keras.Model(
                            inputs=[input_refer, input_pos, input_neg, input_sil],
                            outputs=[embedding_refer, embedding_pos, embedding_neg, embedding_sil]
                            )
    return _network



class Contrastive_Model(keras.Model):

    def __init__(self,
                 Contrastive_network,
                 batch_size,
                 loss_tracker,
                 feature_distance,
                 metric_acc = None,
                 margin = 1,
                 alpha = 1):
        super().__init__()

        self.Contrastive_network = Contrastive_network
        self.margin = margin
        self.alpha = alpha
        self.loss_tracker = loss_tracker
        self.batch_size = batch_size
        self.matric_acc = metric_acc  # not accuracy in the conventional sense

        # define feature distance
        self.feature_distance = feature_distance
        self.distance_func = None
        if self.feature_distance == "Wasserstein":
            self.distance_func = wasserstein_1D
        elif self.feature_distance == "Euclidean":
            self.distance_func = euclidean
        else:
            print("Error: no valid feature distance")


    def _compute_distance(self, inputs):
        # parse data
        (refer, positive, negative, refer_silence) = inputs
        # get wave embeddings
        embeddings = self.Contrastive_network((refer, positive, negative, refer_silence))
        refer_embedding = embeddings[0]
        positive_embedding = embeddings[1]
        negative_embedding = embeddings[2]
        silence_embedding = embeddings[3]

        # calculate distances
        refer_pos_dist = self.distance_func(refer_embedding, positive_embedding)
        refer_neg_dist = self.distance_func(refer_embedding, negative_embedding)
        refer_sil_dist = self.distance_func(refer_embedding, silence_embedding)
        
        # return the distances
        return (refer_pos_dist, refer_neg_dist, refer_sil_dist)


    def _compute_loss(self, refer_pos_dist, refer_neg_dist, refer_sil_dist):
        #-------------
        # Triplet Loss
        #-------------
        loss_1 = refer_pos_dist - refer_neg_dist + self.margin
        loss_1 = tf.maximum(loss_1, 0.0)
        
        loss_2 = refer_pos_dist - tf.math.multiply(refer_sil_dist, self.alpha) + self.margin
        loss_2 = tf.maximum(loss_2, 0.0)

        loss = loss_1 + loss_2

        return loss


    def _compute_acc(self, refer_pos_dist, refer_neg_dist, refer_sil_dist):
        # not accuracy in the conventional sense
        y = [True for i in range(self.batch_size)]
        y_pred = refer_pos_dist < refer_neg_dist
        self.matric_acc.update_state(y, y_pred)
        return self.matric_acc.result()    


    def call(self, inputs):
        (refer_pos_dist, refer_neg_dist, refer_sil_dist) = self._compute_distance(inputs)
        return (refer_pos_dist, refer_neg_dist, refer_sil_dist)
    

    def train_step(self, inputs):

        with tf.GradientTape() as tape:
            # compute the distances
            (refer_pos_dist, refer_neg_dist, refer_sil_dist) = self._compute_distance(inputs)

            # compute the loss
            loss = self._compute_loss(refer_pos_dist, refer_neg_dist, refer_sil_dist)

        # compute gradients and optimize the model
        gradients = tape.gradient(
                                loss,
                                self.Contrastive_network.trainable_variables
                                )
        
        self.optimizer.apply_gradients(zip(gradients, self.Contrastive_network.trainable_variables))

        # update the metrics and return the loss
        self.loss_tracker.update_state(loss)

        metric = self._compute_acc(refer_pos_dist, refer_neg_dist, refer_sil_dist)

        return {"loss": self.loss_tracker.result(),
                "acc": metric,
                }


    def test_step(self, inputs):
        # Compute the distances
        (refer_pos_dist, refer_neg_dist, refer_sil_dist) = self._compute_distance(inputs)

        # Compute the loss
        loss = self._compute_loss(refer_pos_dist, refer_neg_dist, refer_sil_dist)
        
        # Update the metrics and return the loss
        self.loss_tracker.update_state(loss)

        metric = self._compute_acc(refer_pos_dist, refer_neg_dist, refer_sil_dist)


        return {"loss": self.loss_tracker.result(),
                "acc": metric,
                }
    
    @property
    def metrics(self):
        return [self.loss_tracker,
                self.matric_acc]
