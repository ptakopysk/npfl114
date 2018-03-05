#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    # boilerplate
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        # create session
        # threads: default = use all available cpus
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # inside graph is the default
            # graph to use -- adds all stuff into the graph
            # Inputs
            # first shape dimension = batch dimension (for processing  multiple
            # objects at the same time; use "1" if processing one by one)
            # 1 = no of channels (bw/gray=1, rgb = 3...)
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            # Computation
            # linearize the tensor to 1-dim vector (by def NNs do not support
            # tensors, only vectors -- unless we use more advanced stuff)
            flattened_images = tf.layers.flatten(self.images, name="flatten")
            # dense - connect eevrythign to everything
            # activation -- after the summing
            hidden_layer = tf.layers.dense(flattened_images, args.hidden_layer, activation=tf.nn.relu, name="hidden_layer")
            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name="output_layer")
            # axis=1 ... na který ose se to má splácnout (axis=0 pro vektory) --
            # nultá (batches) se má nechat, dál se to má splácnout (takže po
            # obrázkách)
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training (we skipped that for now...)
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.GradientDescentOptimizer(0.03).minimize(loss, global_step=global_step, name="training")

            # Summaries
            # information about training in progress
            # boilerplate...
            # tf.equal -- udělá booleans jestli labels = predictions
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            confusion_matrix = tf.reshape(tf.confusion_matrix(self.labels, self.predictions,
                                                              weights=tf.not_equal(self.labels, self.predictions), dtype=tf.float32),
                                          [1, self.LABELS, self.LABELS, 1])

            # tf.contrib = not part of official stable TF core
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/accuracy", accuracy),
                                               tf.contrib.summary.image(dataset + "/confusion_matrix", confusion_matrix)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    # We are not saving the model, so it gets forgotten when the script ends

    def train(self, images, labels):
        # cíle co chci spustit: training a summaries
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels})

    def evaluate(self, dataset, images, labels):
        # chci jen summaries
        self.session.run(self.summaries[dataset], {self.images: images, self.labels: labels})


if __name__ == "__main__":
    import argparse
    import datetime
    import os

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=100, type=int, help="Size of the hidden layer.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    # identifikace pro tensorboard
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(map(lambda arg:"{}={}".format(*arg), sorted(vars(args).items())))
    )
    # ala mkdir -p
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets("mnist_data/", reshape=False, seed=42)

    # Construct the network
    # this can be slow
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    # e.g. 10x
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
    network.evaluate("test", mnist.test.images, mnist.test.labels)
