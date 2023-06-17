# BIMT: https://kindxiaoming.github.io/pdfs/BIMT.pdf
# a modular approach to regularization and connections
# implemented by finn.archinuk@gmail.com
# v 0.8 ish

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm


from typing import Union

def create_swaps(neuron_index, dims):
    swap_permutations = list()
    for i in range(dims):
        z = np.arange(dims)
        z[i] = neuron_index
        z[neuron_index] = i
        swap_permutations.append(z)
    return swap_permutations

class BIMT_model():
    def __init__(self,
                 hidden_layer_sizes: list,
                 optimizer: tf.keras.optimizers.Optimizer, # default to adam?
                 width: Union[list, float],
                 input_size: int,
                 output_size: int,
                ):
        assert len(hidden_layer_sizes) < 21, 'layers limited to 20 for technical reasons?'
        self.h_layer_sizes = hidden_layer_sizes# [input shape, some list of ints, output shape] # should allow permutations? #this may be unused?
        # 
        self.activations = [tf.keras.activations.swish] * (len(hidden_layer_sizes)) + [tf.keras.activations.softmax]
        self.optimizer   = optimizer # reinitialize after swaps
        self.input_size  = input_size
        self.output_size = output_size
        self.input_patch = tf.eye(input_size) # these get swapped
        self.output_patch = tf.eye(output_size) # these get swapped
        self._layers = [input_size] + hidden_layer_sizes + [output_size]

        if type(width) == list: assert len(width) == len(hidden_layer_sizes)+2, 'since width is a list, must define each layer individually'
        else: width = [width] * (len(hidden_layer_sizes) + 2) #for input and output neurons
        self.embedding_widths = width # < this width ends up being a list

        self.built = False # safety check
        self._build_model() # makes weights and biases
        self._place_neurons() # embeds neurons to a position
        self._find_distances() # calculate distance matrixces for springiness (can be recalculated as desired)
        
    def _build_model(self):
        # build matrices of weights (and biases)
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        trainable_parameters = list()
        
        # hidden layers
        for idx, (former_layer, later_layer) in enumerate(zip(self._layers[:-1],
                                                              self._layers[1:])):
            temp = [tf.Variable(initializer(shape=(former_layer, later_layer))),
                    tf.Variable(initializer(shape=(later_layer,)))]
            trainable_parameters.append(temp)

        self.model = trainable_parameters
        self.built = True # training successful, allows layer functions
        
    def _dist(self, n1, n2):
        n1 = np.array(n1).T
        n2 = np.array(n2).T
        return np.sqrt( np.sum((n1-n2)**2))
    
    def _swapping_neurons(self,
                         incoming_weights,
                         incoming_distances,
                         intermediate_biases,
                         outgoing_weights, 
                         outgoing_distances,
                         top_k = 4):
        # find indices of most significant neurons
        incoming_weights_copy = np.array([np.array(tf.reduce_sum(abs(incoming_weights),axis=0).numpy() + tf.reduce_sum(abs(outgoing_weights),axis=1).numpy()),
                                          np.arange(incoming_weights.shape[1])]) # shape will be (2, n_neurons in layer)
        incoming_weights_rotated = incoming_weights_copy.T[incoming_weights_copy[0].argsort()].T

        # for k most significant neurons, look at their swap potentials and greedily choose best
        for _sn in incoming_weights_rotated.T[-top_k:][::-1]: # _sn for signficant neuron
            _temp_swaps = create_swaps(neuron_index = int(_sn[1]),
                                       dims = incoming_weights_rotated.shape[1]) # creates every combination for the top selected neuron

            springs_test = list()
            for _swap in _temp_swaps:
                _incoming = tf.cast(tf.reduce_sum(np.array(abs(incoming_weights)).T[_swap].T * incoming_distances), dtype=tf.float32)

                _outgoing = tf.cast(tf.reduce_sum(np.array(abs(outgoing_weights))[_swap] * outgoing_distances), dtype=tf.float32)
                springs_test.append((_incoming + _outgoing).numpy())
            _best = np.array(springs_test).argmin()

            incoming_weights = np.array(incoming_weights).T[_temp_swaps[_best]].T
            intermediate_biases = np.array(intermediate_biases).T[_temp_swaps[_best]].T
            outgoing_weights = np.array(outgoing_weights)[_temp_swaps[_best]]

        # 
        incoming_weights = tf.Variable(incoming_weights)
        intermediate_biases = tf.Variable(intermediate_biases)
        outgoing_weights = tf.Variable(outgoing_weights)

        return incoming_weights, intermediate_biases, outgoing_weights

    def _place_neurons(self):
        self.neuron_locations = list()
        for _layer_idx, (_width, _n_neurons)  in enumerate(zip(self.embedding_widths, self._layers)):
            temp_layer = []
            for _y_pos in np.linspace(-_width, _width, _n_neurons):
                temp_layer.append(np.array([_layer_idx, _y_pos]))
            self.neuron_locations.append(np.array(temp_layer))
            
    def _find_distances(self):
        # distances for each neuron (eventually break this and positions into another module for increased customization)?
        
        self._distances = list()
        
        # for each pair of layers
        for _in_layer, _out_layer in zip(self.neuron_locations[:-1], self.neuron_locations[1:]):
            temp = np.zeros((len(_in_layer), len(_out_layer)))
            for idx0, _n0 in enumerate(_in_layer):
                for idx1, _n1 in enumerate(_out_layer):
                    temp[idx0, idx1] = self._dist(_n0, _n1)
            self._distances.append(temp)

    def predict(self, data):
        assert self.built, 'build model before prediction (otherwise no weights)'
        assert data.shape[1] == self.input_size
        data = tf.cast(data, dtype=tf.float32)

        running_data = tf.matmul(data, self.input_patch) # apply input patch

        for _act, _layer in zip(self.activations, self.model): # apply all hidden layers
            running_data = _act(tf.add(tf.matmul(running_data, _layer[0]), _layer[1]))
            
        running_data = tf.matmul(running_data, self.output_patch) # apply output patch
        return running_data
    
    @property
    def trainable_parameters(self): # used in training
        return [_params for _layer in self.model for _params in _layer]
        
    def train(self,
              training_data_input,
              training_data_output,
              val_input,
              val_output,
              loss_fn = 'mse', # or cat_cross
              n_epochs = 10,
              top_k=5,
              learning_rate = 1e-2,
              l1_coeff = 1e-3, # l1 tries to remove unused weights
              reg_coeff = 1e-3 # regularizer on spring and bias # (consider separating these?)
             ):

        self.loss_log = {'main_loss':[], 'l1':[], 'spring':[], 'bias':[], 'total':[], 'val_main_loss':[]}
        
        assert loss_fn in ['mse','cat_cross'], 'loss function not yet implemented'
        
        if loss_fn == 'mse': _internal_loss_func = tf.keras.losses.MSE() # < this seems to cause some issues
        if loss_fn == 'cat_cross': _internal_loss_func = tf.keras.losses.CategoricalCrossentropy()
        
        for k in tqdm(range(n_epochs)):
            # reinitialize optimizer
            opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            for _batch_in, _batch_out in zip(training_data_input, training_data_output):
                temp_loss_log = {'main_loss':[], 'l1':[], 'spring':[], 'bias':[], 'total':[]}

                with tf.GradientTape() as tape:
                    # prediction cost
                    pred_cost = tf.reduce_mean(_internal_loss_func(_batch_out, self.predict(_batch_in)))

                    # l1 cost
                    l1_cost = tf.reduce_sum(tf.math.abs(self.model[0][0]))
                    for _weights in self.model[1:]:
                        l1_cost += tf.reduce_sum(tf.math.abs(_weights[0]))

                    # spring cost
                    # (spring cost only calculates springiness to later layers, which is probably a sufficient approximation)
                    spring_cost = tf.reduce_sum(tf.math.multiply(self._distances[0], tf.math.abs(self.model[0][0])))
                    for _weights, _distance in zip(self.model[1:], self._distances[1:]):
                        spring_cost += tf.reduce_sum(tf.math.multiply(_distance, tf.math.abs(_weights[0])))
                        
                    # bias costs
                    bias_cost = tf.reduce_sum(tf.math.abs(self.model[0][1]))
                    for _bias in self.model[1:]:
                        bias_cost += tf.reduce_sum(tf.math.abs(_weights[1]))

                    total = pred_cost + l1_coeff * l1_cost + reg_coeff * (spring_cost + bias_cost)

                # log these loss components
                temp_loss_log['main_loss'].append(pred_cost.numpy().mean())
                temp_loss_log['l1'].append(l1_coeff * l1_cost.numpy().mean())
                temp_loss_log['spring'].append(reg_coeff * spring_cost.numpy().mean())
                temp_loss_log['bias'].append(reg_coeff * bias_cost.numpy().mean())
                temp_loss_log['total'].append(total.numpy().mean())

                grad = tape.gradient(total,   self.trainable_parameters)
                opt.apply_gradients(zip(grad, self.trainable_parameters))

            # save epoch loss values
            self.loss_log['main_loss'].append(np.array(temp_loss_log['main_loss']).mean())
            self.loss_log['l1'].append(np.array(temp_loss_log['l1']).mean())
            self.loss_log['spring'].append(np.array(temp_loss_log['spring']).mean())
            self.loss_log['bias'].append(np.array(temp_loss_log['bias']).mean())
            self.loss_log['total'].append(np.array(temp_loss_log['total']).mean())
            # calculate main_loss for validation set
            pred_cost = tf.reduce_mean(_internal_loss_func(val_output, self.predict(val_input)))
            self.loss_log['val_main_loss'].append(pred_cost.numpy().mean())
            
            ''' ------------- SWAP NEURON LOCATIONS ------------ '''
            # swap input patch
            self.input_patch, _, self.model[0][0] = self._swapping_neurons(incoming_weights = self.input_patch,
                                                                           incoming_distances = tf.ones(self.input_size), #placeholder, distances don't matter on this layer
                                                                           intermediate_biases = tf.zeros(self.input_size), # placeholder, no bias in swap
                                                                           outgoing_weights = self.model[0][0],
                                                                           outgoing_distances = self._distances[0],
                                                                           top_k = top_k)
            # swap hidden layers
            for idx, ((_weight0, _bias0), (_weight1, _unused), _d0, _d1) in enumerate(zip(self.model[:-1],     # how did this get so gross
                                                                                          self.model[1:],
                                                                                          self._distances[:-1],
                                                                                          self._distances[1:])):

                self.model[idx][0], self.model[idx][1], self.model[idx+1][0] = self._swapping_neurons(incoming_weights = _weight0,
                                                                                                      incoming_distances = _d0,
                                                                                                      intermediate_biases = _bias0,
                                                                                                      outgoing_weights = _weight1,
                                                                                                      outgoing_distances = _d1,
                                                                                                      top_k = top_k)
            # swap outputs
            self.model[-1][0], self.model[-1][1], self.output_patch = self._swapping_neurons(incoming_weights = self.model[-1][0],
                                                                                             incoming_distances = self._distances[-1],
                                                                                             intermediate_biases = self.model[-1][1],
                                                                                             outgoing_weights = self.output_patch,
                                                                                             outgoing_distances = tf.ones(self.output_size),
                                                                                             top_k = top_k)



            ''' ------------ IF YOU WANT TO DO SOMETHING WEIRD, DOWN HERE IS THE PLACE --------------- '''
            # weird like puting this thing in networkx and find positions of neurons that way
            # instead of a fixed linearized positioning scheme. (don't forget to recalculate neuron distances)

            ''' ------------ END OF WEIRD STUFF --------------- '''
        
    def plot_lr_curves(self):
        assert hasattr(self, 'loss_log'), 'train model first'
        fig,ax = plt.subplots(1, len(self.loss_log.keys())-1, figsize=(18,4))

        for idx, _l in enumerate(self.loss_log.keys()):
            if _l == 'val_main_loss': idx = 0 # place validation of main loss on same figure
            ax[idx].plot(self.loss_log[_l], label=_l)

        for idx in range(len(self.loss_log.keys())-1): ax[idx].grid(); ax[idx].legend();
        plt.tight_layout()
        plt.show()
        
    def plot_weights(self):
        def create_norm(data):
            data = np.array(data)
            return mpl.colors.TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())

        fig,ax = plt.subplots(1,len(self.model)+2,figsize=(14,4))

        panel = ax[0].imshow(self.input_patch, cmap='plasma')
        plt.colorbar(panel, ax=ax[0])

        for i, (w,b) in enumerate(self.model):
            panel = ax[i+1].imshow(w, norm=create_norm(w), cmap='seismic') # "+1" to offset from the input patch
            plt.colorbar(panel, ax=ax[i+1])

        panel = ax[-1].imshow(self.output_patch, cmap='plasma')
        plt.colorbar(panel, ax=ax[-1])
        plt.tight_layout()
        plt.show()
        
    def plot_connections(self,
                         input_names: list = None, # feature names
                         output_names: list = None, # targets
                         edge_threshold = 1e-1):
        
        # make labels for features
        if input_names == None: input_names = [f'x{i}' for i in range(self.input_size)]
        if output_names == None: output_names = [f'y{i}' for i in range(self.output_size)] 
        
        # plot neuron locations
        for i in self.neuron_locations:
            plt.scatter(*i.T)

        # plot edges
        for _prev_idx, _later_idx in zip(np.arange(len(self.model)), np.arange(1,len(self.model)+1)):

            for _idx1, _pos1 in enumerate(self.neuron_locations[_prev_idx]):
                for _idx2, _pos2 in enumerate(self.neuron_locations[_later_idx]):
                    _weight = self.model[_prev_idx][0][_idx1][_idx2]
                    if tf.abs(_weight) > edge_threshold: # technically not required, but way faster if not plotting everything
                        if _weight > 0: color='b'
                        else: color='r'
                        plt.plot([_pos1[0],_pos2[0]],
                                 [_pos1[1],_pos2[1]],
                                 ls='--',linewidth=tf.abs(_weight),c=color)

        # feature names
        sorted_intput_args = np.array(input_names)[np.argmax(self.input_patch,axis=0)]
        for _argument, _pos in zip(sorted_intput_args,
                                   self.neuron_locations[0]):
            plt.text(*_pos, _argument)

        sorted_output_args = np.array(output_names)[np.argmax(self.output_patch,axis=0)]
        for _argument, _pos in zip(sorted_output_args,
                                   self.neuron_locations[-1]):
            plt.text(*_pos, _argument)
        plt.show()
        
    def vis_evaluate(self, test_input, test_output):
        assert test_output.shape[1] == 2, 'this visual evaluation only work on 2D outputs'
        
        pred_values = self.predict(test_input)
        plt.figure()
        plt.scatter(test_output[:,0], test_output[:,1],label='real')
        plt.scatter(pred_values[:,0], pred_values[:,1],label='pred')
        plt.grid()
        plt.legend()
        plt.show()
        
        
        
        
if __name__ == '__main__':
    
    print('\n running using toy dataset (v0.8)')
    print(f'using tf version {tf.__version__}')
    
    # a sample dataset
    def gen_samples(n_samples):
        a = np.random.normal(0,1,size=(n_samples,4))
        return (a, np.array([a[:,0]*a[:,3] + a[:,1]*a[:,2], a[:,0]*a[:,3] - a[:,1]*a[:,2]]).T)

    my_model_new = BIMT_model(input_size = 4,
                               output_size = 2,
                               hidden_layer_sizes = [12,10,6],
                               optimizer = 'adam',
                               width = [3,6,5,3,2]) # how much space neurons take 

    # create training dataset
    (train_data_in, train_data_out) = gen_samples(39936) # this value fits nicely with batch sizes of 128.
    # training dataset shape is expected to be: (number of batches, batch size, data shape)
    train_data_in = train_data_in.reshape(-1, 128, 4)
    train_data_out = train_data_out.reshape(-1, 128, 2)

    # validation set
    (val_train_data_in, val_train_data_out) = gen_samples(200) # this value fits nicely with batch sizes of 128.
    # training dataset shape is expected to be: (number of batches, batch size, data shape)
    val_train_data_in = val_train_data_in.reshape(-1, 4)
    val_train_data_out = val_train_data_out.reshape(-1, 2)

    
    # train the model
    my_model_new.train(train_data_in,
                       train_data_out,
                       val_train_data_in,
                       val_train_data_out,
                       n_epochs = 10,
                       top_k=5,
                       learning_rate = 1e-3,
                       l1_coeff = 1e-5, # l1 tries to remove unused weights
                       reg_coeff = 1e-5 # regularizer on spring and bias # (consider separating these?)
                      )

    
    # plot results
    my_model_new.plot_lr_curves()
    
    my_model_new.plot_weights()
    
    my_model_new.vis_evaluate(*gen_samples(100))
    
    my_model_new.plot_connections()