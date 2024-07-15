import torch
from tqdm.auto import tqdm
from CNN_Version2.learning_rate import LearningRateDecay
from dataset import dataset
from CNN_Version2.conv2d import Conv2D
from CNN_Version2.dense_layer import Dense
from CNN_Version2.flatten import Flatten
from VGG16.pooling import Pooling2D
from loss import CrossEntropyLoss


class CNN:

    def __init__(self, layers=None):
        self.cost = None
        self.layer_name = None
        self.d = None
        self.architecture = None
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        self.network_architecture_called = False

    def add(self, layer):
        self.layers.append(layer)

    def Input(self, input_shape):
        self.d = input_shape
        self.architecture = [self.d]
        self.layer_name = ["Input"]

    def network_architecture(self):
        for layer in self.layers:
            if layer.__class__.__name__ == 'Conv2D':
                if layer.input_shape_x is not None:
                    self.Input(layer.input_shape_x)
                layer.get_dimensions(self.architecture[-1])
                self.architecture.append(layer.output_shape)
                self.layer_name.append(layer.__class__.__name__)
            elif layer.__class__.__name__ in ['Flatten', 'Pooling2D']:
                layer.get_dimensions(self.architecture[-1])
                self.architecture.append(layer.output_shape)
                self.layer_name.append(layer.__class__.__name__)
            elif layer.__class__.__name__ == 'Dense':
                self.architecture.append(layer.out_features)
                self.layer_name.append(layer.__class__.__name__)
            else:
                self.architecture.append(self.architecture[-1])
                self.layer_name.append(layer.__class__.__name__)

        self.layers = [layer for layer in self.layers if layer is not None]
        try:
            idx = model.layer_name.index("NoneType")
            del model.layer_name[idx]
            del model.architecture[idx]
        except:
            pass

    def summary(self):
        if not self.network_architecture_called:
            self.network_architecture()
            self.network_architecture_called = True
        len_assigned = [45, 26, 15]
        count = {'Dense': 1, 'Activation': 1, 'Input': 1,
                 'BatchNormalization': 1, 'Dropout': 1, 'Conv2D': 1,
                 'Pooling2D': 1, 'Flatten': 1}

        col_names = ['Layer (type)', 'Output Shape', '# of Parameters']

        print("Model: CNN")
        print('-' * sum(len_assigned))

        text = ''
        for i in range(3):
            text += col_names[i] + ' ' * (len_assigned[i] - len(col_names[i]))
        print(text)

        print('=' * sum(len_assigned))

        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        for i in range(len(self.layer_name)):
            # layer name
            layer_name = self.layer_name[i]
            name = layer_name.lower() + '_' + str(count[layer_name]) + ' ' + '(' + layer_name + ')'
            count[layer_name] += 1

            # output shape
            try:
                out = '(None, '
                for n in range(len(model.architecture[i]) - 1):
                    out += str(model.architecture[i][n]) + ', '
                out += str(model.architecture[i][-1]) + ')'
            except:
                out = '(None, ' + str(self.architecture[i]) + ')'

            # number of params
            if layer_name == 'Dense':
                h0 = self.architecture[i - 1]
                h1 = self.architecture[i]
                if self.layers[i - 1].use_bias:
                    params = h0 * h1 + h1
                else:
                    params = h0 * h1
                total_params += params
                trainable_params += params
            elif layer_name == 'BatchNormalization':
                h = self.architecture[i]
                params = 4 * h
                trainable_params += 2 * h
                non_trainable_params += 2 * h
                total_params += params
            elif layer_name == 'Conv2D':
                layer = self.layers[i - 1]
                if layer.use_bias:
                    add_b = 1
                else:
                    add_b = 0
                params = ((layer.Nc * layer.kh * layer.kw) + add_b) * layer.out_channels
                trainable_params += params
                total_params += params
            else:
                # Pooling, Dropout, Flatten, Input
                params = 0
            names = [name, out, str(params)]

            # print this row
            text = ''
            for j in range(3):
                text += names[j] + ' ' * (len_assigned[j] - len(names[j]))
            print(text)
            if i != (len(self.layer_name) - 1):
                print('-' * sum(len_assigned))
            else:
                print('=' * sum(len_assigned))

        print("Total params:", total_params)
        print("Trainable params:", trainable_params)
        print("Non-trainable params:", non_trainable_params)
        print('-' * sum(len_assigned))

    def compile(self, cost_type, optimizer_type):
        self.cost = CrossEntropyLoss()
        self.cost_type = cost_type
        self.optimizer_type = optimizer_type

    def initialize_parameters(self):
        if self.network_architecture_called == False:
            self.network_architecture()
            self.network_architecture_called = True
        # initialize parameters for different layers
        for i, layer in enumerate(self.layers):
            if layer.__class__.__name__ in ['Dense', 'Conv2D']:
                layer.initialize_parameters(self.architecture[i], self.optimizer_type)
            elif layer.__class__.__name__ == 'BatchNormalization':
                layer.initialize_parameters(self.architecture[i])

    def fit(self, train_dataloader, test_dataloader, epochs=10, batch_size=5, lr=1, X_val=None, y_val=None, verbose=1,
            lr_decay=None, **kwargs):

        self.history = {'Training Loss': [],
                        'Validation Loss': [],
                        'Training Accuracy': [],
                        'Validation Accuracy': []}

        iterations = 0
        self.m = batch_size
        self.initialize_parameters()

        for epoch in tqdm(range(epochs)):
            cost_train = 0
            num_batches = 0

            print('\nEpoch: ' + str(epoch + 1) + '/' + str(epochs))

            for batch, (X_batch, y_batch) in enumerate(train_dataloader):

                Z = X_batch.clone()

                # feed-forward
                for layer in self.layers:
                    Z = layer.forward(Z)

                # calculating the loss
                cost_train += self.cost(Z, y_batch)

                # calculating dL/daL (last layer backprop error)
                dZ = self.cost.backward(Z, y_batch)

                # backpropagation
                for layer in self.layers[::-1]:
                    dZ = layer.backpropagation(dZ)

                # Parameters update
                for layer in self.layers:
                    if layer.__class__.__name__ in ['Dense', 'BatchNormalization', 'Conv2D']:
                        layer.update(lr, self.m, iterations)

                # Learning rate decay
                if lr_decay is not None:
                    lr = lr_decay(iterations, **kwargs)

                num_batches += 1
                iterations += 1

            cost_train /= num_batches

            # printing purpose only (Training Accuracy, Validation loss and accuracy)

            text = 'Training Loss: ' + str(round(cost_train, 4)) + ' - '
            self.history['Training Loss'].append(cost_train)

            # training accuracy

            if self.cost_type == 'cross-entropy':
                train_acc = (Z.argmax(dim=1) == y_batch).sum().item() / len(y_batch)
                text += 'Training Accuracy: ' + str(round(train_acc, 4))
                self.history['Training Accuracy'].append(train_acc)
            else:
                text += 'Training Accuracy: ' + str(round(cost_train, 4))
                self.history['Training Accuracy'].append(cost_train)

            if X_val is not None:
                cost_val, accuracy_val = self.evaluate(X_val, y_val, batch_size)
                text += ' - Validation Loss: ' + str(round(cost_val, 4)) + ' - '
                self.history['Validation Loss'].append(cost_val)
                text += 'Validation Accuracy: ' + str(round(accuracy_val, 4))
                self.history['Validation Accuracy'].append(accuracy_val)

            if verbose:
                print(text)
            else:
                print()


if __name__ == "__main__":
    train_dataloader, test_dataloader = dataset()
    input_shape, output_dim = next(iter(train_dataloader))[0].shape, 10
    model = CNN()

    model.add(model.Input(input_shape=input_shape))

    model.add(Conv2D(1, out_channels=32, kernel_size=(5, 5), padding='same', activation_type="relu",
                     weight_initializer_type='he_normal'))

    model.add(Pooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # model.add(Dropout(0.4))

    model.add(Dense(output_dim, activation_type="softmax"))
    batch_size = next(iter(train_dataloader))[0].shape[0]
    epochs = 10
    lr = 0.05

    model.compile(cost_type="cross-entropy", optimizer_type="adam")

    LR_decay = LearningRateDecay()

    model.fit(train_dataloader, test_dataloader, epochs=epochs, batch_size=batch_size, lr=lr,
              verbose=1, lr_decay=LR_decay.constant, lr_0=lr)

    model.summary()
