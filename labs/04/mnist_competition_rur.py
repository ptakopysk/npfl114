#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# ygenerovanej extra trénink set
# ty data maj mnist formát takže když nacpu ty data do nějaký složky a řeknu že
# tam jsou tak ten kód to tam najde a načte to místo skutečnýho mnistu

# output = labels for test set (digits 0-9)

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # TODO: Construct the network and training operation.

            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Computation
            # TODO: Add layers described in the args.cnn. Layers are separated by a comma and can be:
            # - C-filters-kernel_size-stride-padding: Add a convolutional layer with ReLU activation and
            #   specified number of filters, kernel size, stride and padding. Example: C-10-3-1-same
            # - M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
            # - F: Flatten inputs --- tim se ztratí shape takže už nejde dělal
            # cnn ale zato jde dělat densely connected (někdy to tam přijde
            # takže nakonec to bude v poho)
            # - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
            # - D-rate: Add dropout with given rate
            # Store result in `features`.

            layer = self.images
            for definition in args.cnn.split(','):
                parameters = definition.split('-')
                if parameters[0] == 'C':
                    layer = tf.layers.conv2d(layer,
                            int(parameters[1]),
                            int(parameters[2]),
                            int(parameters[3]),
                            parameters[4],
                            activation=tf.nn.relu)
                elif parameters[0] == 'M':
                    layer = tf.layers.max_pooling2d(layer,
                            int(parameters[1]),
                            int(parameters[2]))
                elif parameters[0] == 'F':
                    layer = tf.layers.flatten(layer)
                elif parameters[0] == 'R':
                    layer = tf.layers.dense(layer, int(parameters[1]), activation=tf.nn.relu)
                elif parameters[0] == 'D':
                    layer = tf.layers.dropout(layer, rate=float(parameters[1]),
                            training=self.is_training)
                else:
                    assert False, "invalid definition " + definition

            output_layer = tf.layers.dense(layer, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            learning_rate = args.learning_rate
            if args.learning_rate_final:
                # compute parameters
                decay_rate = (args.learning_rate_final/args.learning_rate)**(1/(args.epochs-1))
                learning_rate = tf.train.exponential_decay(args.learning_rate,
                        global_step, args.batches_per_epoch, decay_rate, staircase=True,
                        name="learning_rate")
            self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]],
            {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, images, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]],
            {self.images: images, self.labels: labels, self.is_training: False})
        return accuracy
       
    def predict(self, dataset, images, labels):
        predictions = self.session.run(self.predictions,
            {self.images: images, self.labels: labels, self.is_training: False})
        return predictions
       

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", default="C-10-3-2-same,M-3-2,F,R-100", type=str, help="Description of the CNN architecture.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.");
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.");
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42)
    args.batches_per_epoch = mnist.train.num_examples // args.batch_size

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        accuracy = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        print("{:.2f}".format(100 * accuracy))

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    predictions = network.predict("test", mnist.test.images, mnist.test.labels)
    test_labels = predictions

    for label in test_labels:
        print(label)
