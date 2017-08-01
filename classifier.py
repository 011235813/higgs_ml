import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import networks
import argparse


class Classifier():

    def __init__(self, num_layers, nonlinearity1, nonlinearity2,
                 nonlinearity3, n_inputs, n_hidden1, n_hidden2,
                 n_hidden3, n_outputs, lr, batch_size,
                 input_file, log_dir, gpu, test_mode):

        self.filename = 'HIGGS.csv'
        self.file_length = 11000000        

        self.lr = lr
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.test_mode = test_mode

        self.map_str_nonlinearity = {'relu':tf.nn.relu, 'tanh':tf.nn.tanh}

        self.examples, self.labels = self.input_pipeline()

        self.create_network(num_layers, nonlinearity1, nonlinearity2,
                             nonlinearity3, n_inputs, n_hidden1,
                             n_hidden2, n_hidden3, n_outputs)

        # TODO: need to create test set

        if not self.test_mode:
            self.train_op = self.create_training_method()

        # for recording the entire network weights
        self.saver = tf.train.Saver()

        if gpu:
            session_config = tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=True)
            session_config.gpu_options.allow_growth = True
            self.session = tf.InteractiveSession(config=session_config)
        else:
            self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())            


    def create_network(self, num_layers, nonlinearity1, nonlinearity2,
                       nonlinearity3, n_inputs, n_hidden1,
                       n_hidden2, n_hidden3, n_outputs):

        #self.vec_input = tf.placeholder(dtype=tf.float64,
        #                                shape=[None, n_inputs],
        #                                name='vec_input')
        self.is_train = tf.placeholder(dtype=tf.bool,
                                       name='is_train')
        
        if num_layers == 3:
            self.y = networks.hidden3_bn(self.examples, n_hidden1,
                                         n_hidden2, n_hidden3,
                                         n_outputs, self.map_str_nonlinearity[nonlinearity3], self.is_train)


    def read_from_csv(self, filename_queue):
        reader = tf.TextLineReader(skip_header_lines=0)
        key, value = reader.read(filename_queue)
        record_defaults = [[0.0]] * 29
        columns = tf.decode_csv(value, record_defaults)
        features = tf.stack(columns[1:])
        label = tf.stack(columns[0:1])
        return features, label
    
    
    def input_pipeline(self, num_epochs=None):
        filename_queue = tf.train.string_input_producer([self.filename])
        example, label = self.read_from_csv(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self.batch_size
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=self.batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        label_batch_int = tf.squeeze(tf.cast(label_batch, dtype=tf.int32))
        return example_batch, label_batch_int


    def create_training_method(self):
    
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(self.y),dtype=tf.int32), self.labels), dtype=tf.float32))
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.y))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return train_op
    

    def main(self):
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        count = 0 
        try:
            # while not coord.should_stop():
            while count < 3:
                example_batch, label_batch = self.session.run([self.examples,self.labels])
                print 'Examples', example_batch
                print 'Labels', label_batch
                # self.session.run(self.train_op)
                count += 1
        except tf.errors.OutOfRangeError:
            print("Done training")
            print count
        finally:
            coord.request_stop()
    
        coord.join(threads)
        self.session.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", default=1,
                        type=int, choices=[1,2,3],
                        help="number of hidden layers")
    parser.add_argument("--nonlinearity1", default='relu',
                        type=str, help="nonlinear function for hidden layer 1")
    parser.add_argument("--nonlinearity2", default='relu',
                        type=str, help="nonlinear function for hidden layer 2")
    parser.add_argument("--nonlinearity3", default='relu',
                        type=str, help="nonlinear function for hidden layer 3. If three layers, then nonlinearity for first two layers is fixed as relu")
    parser.add_argument("--n_inputs", default=28,
                        type=int, help="dimension of input layer")    
    parser.add_argument("--n_hidden1", default=128,
                        type=int, help="width of hidden layer 1")
    parser.add_argument("--n_hidden2", default=128,
                        type=int, help="width of hidden layer 2")
    parser.add_argument("--n_hidden3", default=128,
                        type=int, help="width of hidden layer 3")
    parser.add_argument("--lr", default=1e-3,
                        type=float, help="optimizer learning rate")
    parser.add_argument("--batch_size", default=64,
                        type=int, help="batch size")
    parser.add_argument("--input_file", default='HIGGS.csv',
                        type=str, help="location to save network, tensorboard and results")
    parser.add_argument("--log_dir", default=None,
                        type=str, help="location to save network, tensorboard and results")
    parser.add_argument("--gpu", action="store_true",
                        help="if flag is set, then configures tensorflow session for GPU")
    parser.add_argument("--test", action="store_true",
                        help="if flag is set, then reads network from log_dir and tests on test data")
    args = parser.parse_args()                          


    c = Classifier(args.num_layers, args.nonlinearity1,
                   args.nonlinearity2, args.nonlinearity3,
                   args.n_inputs, args.n_hidden1, args.n_hidden2,
                   args.n_hidden3, 1, args.lr, args.batch_size,
                   args.input_file, args.log_dir, args.gpu, args.test)

    c.main()
