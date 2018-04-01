import loader
import neural_net

training_data, validation_data, test_data = loader.load_data_wrapper()

net = neural_net.Network([784, 64, 16, 10])

#SGD(training_data, epochs, mini_batch_size, eta, test_data)
net.SGD(training_data, 30, 10, 3.5, test_data=test_data)
