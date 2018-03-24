# coding=utf-8

source_1 = """#!/usr/bin/env python3
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
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

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
                    assert False, \"invalid definition \" + definition

            output_layer = tf.layers.dense(layer, self.LABELS, activation=None, name=\"output_layer\")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")
            global_step = tf.train.create_global_step()
            learning_rate = args.learning_rate
            if args.learning_rate_final:
                # compute parameters
                decay_rate = (args.learning_rate_final/args.learning_rate)**(1/(args.epochs-1))
                learning_rate = tf.train.exponential_decay(args.learning_rate,
                        global_step, args.batches_per_epoch, decay_rate, staircase=True,
                        name=\"learning_rate\")
            self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name=\"training\")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries[\"train\"]],
            {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, images, labels):
        accuracy, predictions, _ = self.session.run([self.accuracy,
            self.predictions, self.summaries[dataset]],
            {self.images: images, self.labels: labels, self.is_training: False})
        return (accuracy, predictions)
       

if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--cnn\", default=\"C-10-3-2-same,M-3-2,F,R-100\", type=str, help=\"Description of the CNN architecture.\")
    parser.add_argument(\"--batch_size\", default=50, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=10, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--learning_rate\", default=0.001, type=float, help=\"Initial learning rate.\");
    parser.add_argument(\"--learning_rate_final\", default=None, type=float, help=\"Final learning rate.\");
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(\"mnist-gan\", reshape=False, seed=42)
    args.batches_per_epoch = mnist.train.num_examples // args.batch_size

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        accuracy = network.evaluate(\"dev\", mnist.validation.images,
                mnist.validation.labels)[0]
        print(\"{:.2f}\".format(100 * accuracy))

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    predictions = network.evaluate(\"test\", mnist.validation.images,
            mnist.validation.labels)[1]
    test_labels = predictions

    for label in test_labels:
        print(label)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;3v%rb6o%;o(hLXI7{R|qXF8eM*A<v`XKyGRLe{Hn@C6f$YVlUqh)H3N*!{WweSgTD8qAk5v3dO4%TATp^Bu_8jXm%yM;V;n;KnSO4b7}O({O4pNVnBA!yCNpoShnRw5dPT30$Wa^9|_`B2?@Sy|y%NZSYCAiNh>n3?n8jSoy|<+DgzY_~vs2;BL-sv)a&Gd|p}wfPcg{GIusQ1<gQEQE0_s84Avyd&<pZEyFqESo162V-0oWI2`W&H3ML?&Gzd5Kt8kRfrt2pXEuQO5JYdpFTC5sI^+BBiik9Ta!D~OZ=oA8}X)_FHHx1Jw6pIq_~NPFi+nsyJf6})tjSXklH%#!kckQ1y*;Hb+Jp97Y>`c@PWv{RxR$9#&Ub4E=qCIsp0W~H`Ck{`rZ`*ap&sK6Yi-3D#;fZb=#t+otA6%1>Pg?+i(K&3RpLm8G*u^wS!Tk=!lB}Hy_qdTu%01W7VS|F}hJFHZ(s<0J0iQ2V!1@2Q9&$R&PW(%5~W()w3<3lB?7Qp(ZtIdk`zij2%z)8d5M{P3ha7voY|O1Y3ULVPq#(AZ^!wVpUe85<ZzJ&{R6tz@-#zdOOhp=7b8G?I+I*6$R|6x%np?hO{toe{OMt%K+d@Xp<<g@?OZrRaK@b;Rwr_&#>#Jfb(nkRpmA$#86pHD&yZ^V=h?n&dPt4u9oR4-QAtGWR`}!nvzJHoKbnQg&dnp%<AITu(VGHCrH?_tk0mjXaZn9<&cW>noS0N7T%c+P13d4u~*^$<3Y~u^w?F(6bX$eRHD;v1&VE?>=DiTQ2DNTAefHxUUR!J)$PwGAK)SH1vxUs4>TcyRB~(-mWEi=O;d}x7+`Aev#C(RXHNBpBAtrQwDBR(6ts`2t%iIA1Hk{#OvSqn$D`t4{ZP>#IFWRHX9Oh0aoOQ7oTx!Az@MV0Fn`O>epCJ8W;ehBSiA{x=x3Nj`{Ja5hqh_f8z3e0NKLWZ2ED1&39yt-H3jUlXPs2G^VqgZ^~<^#U76ooV$ziVV<gKWe3y9>_a0CG;(bsTysS<Kd@MXMeG-jctRZp0v-8fMG{qSVXXl|W0bdnmUPfA7hnm>(Pwi@t$T!D}h_mVkX||~mO7W5L+CN58OY)VmN7Q8<BtB`xHCRKuBYH#M5b3)1z*8Ei3LI_WQ1(G-oB}+eL6%^llfdXak#(uNbt8WFQcZpQmx!sI#!1C1a62)iDW_%)1}z?MFWfb!iEw+vMSr~_4B5{OFjyK5K4OT}6a7zB>oMcd9>tKWxcT$1qxK*9p@I|+0}UzyVd1K4EJF(t@^9q2K6)MXnU*xD6Z4SmAw{0{=q<ZDw{_QiEUN@EW0^zS_Js*Fxn2w>D40o&(Bj%4>^2wsxmZ!?UZ_e%GstPon>HD<>6yGO>Rx7Jt08tW`jMZdwpSiqdm;sie@wdP<h~SOVpDZhq=AK$Hn0tnCh)VEJ=uaAkxnKgy;PhBeeW?X<w!a@d7sdMPkpq?2TiA7SlX2h7ws%AZj}t0R1iUEQ~(}Dp6u*0ubZ_c5@uiJb}szU<wHIH(>x`E79SLKr&_9<!UrtFzXhvu2eAcvL_e>np@$b^EXwrvVcIR0UcZ!R+Lr(=R`u&XY6eHJ>M5fs)GA(KYwt#x(jJhq|Gl>;IKLH-hK4tu8Xa!Rim0!g#yX~Mvf0Gki@_=rH-=xV$Nkd};HN{kmU*|fE?%A^AViQe%x-k7ul=My?k`^kf)%njWo<X65sS)r)R@W+X9F6=yU($Mt&IRVK0hIRDA;KSUvK96Ih1K3i~*w};Fnmlc=!DrPJ&0Ig#=$(`wmSzF3!P88*cNc33p!bf)$kNK*p`OhPyQ+>`#d~3!w+++N8emZgkUUYo|@ngNyDmk^z<K?)H(_hSdK;JizErE7L5<BzyPmAw6koE2%f}GJ})Wg$D(cCH4H^J~K=OS@=>oweBl59yK&l07Q6^+}gRn`!spDPg7_1tY4YQ-5v|joABo_4$y@q*mqQCYR$K{1u)#4TR-@*+2D!%Q-s}|z1gV@L5?<P0s)J#dC)RKYRV*0UV<h-6MtfD+s`)62wlQuGIyL@A4C%GRmy`QwC()GDzv=3o&!mR2W)1*50iB;C#^m)sMzC0Z7-I17wN8J>8ip1l<UtzhfkA*rV*W+QuXH;PlZP2NtcW}-+ev2PbY;6391N*<@B;mEXSM@3k~&n(iKCi3pZ*ILKyS70ReC3`4ieKw^r`Oz~0URr(7F>#PH%4wNYH0`d*?2{&3z9B$jFD!%4Dc>~LZf=#E&70=B0^!QYs@;|?WEpHppz#S$xxEiXWV6QNEP8&i)SGWlf3ixf{%8UNQ9b`adP-gk(zC17b^R~_jt!|(4eRcfIZ8rmJG_+O!lUS|@S{RlQDIAftVhzwG@5ML^WW?|`D+_dW%1rUn#*Rue#0M%r<Jok=1p{u`27&CPsB6U`_%InMrSUDx*TlH_LxiK*fcUP)NP%;9K4c6Wp(07%cI+cOVYMAH!H2)&nc|l~+XqOZVw!YnXQXWOFX11S_w|W@RQ6^ey(DM9u|AkgT6Ztx(e!AuDasU>~;Ogj8OjAdtCe@DAXd}u%$#T%}%_+Yww6x|~A3Pa^dx{9sQYaeOKs`OxBe!YaFtG=?sWvR1b_48N2n22HPp!9Kj`=nC#C-PGnxIOoY~?!26hn`$VSlw9>dMv;{(()ww7?*{@MugO#rM>@w4@J%C2DT^W0Sb7+xq9)e|8mejXC7y^aub1k(J_4IP=FtBMJA~MXI|_PyvU^4Vb0Cj@GCO8qttoP4pt8bwPw6$iYpp0xbJN0qg9BFHCjcwkN7YaV^RJhO`S-Y=#?(gUnT+d*~@$ICF=aw8LaXYQuz+q50)f3BoqY<fxpEoKDNEc@{PutQ!0sD+Jq!u&gq*&5mi`xnzA98cVnLo%p*(OxFkf=gs11cweg{`dojm#g3?J$@HCF^W+7xF;6+T8lS^FovB$PUkZBRj+T^$l#OT=YtB68otzaIZ1VVUd%V|;T4)s}^lf*_E+?s#%GUwm9K;&s3Q)_=X3oFVD}wl+ULBDISMriw0vIJz@}G8?uE?_<C|lK0t^t)`;}L4t6An9dJvh@A&R<!F&Ykg`KMTmM@MP)6pivn|_f^xX@WDK<wpq}F6N#$R3v3{N$Z(Cc5u8<d7oJV|g0N#t+=bO6Yfzw8;a^e}ZX@NeJGtMXP{KVFrzJLSkMQs0*XRGA`j9bc-a=&kHEimmzRvW$kOL2u;IHtDtz*&`ax~GrV<6Pnc0B<Ja-iBHU(Rz)+Jzav<&QbuW_fgcOZ*27p|MMgfEVLLhS!}JEPG}eU&mN%k+-kj`QY6n_VE*2qFSFa3WI5Kb?W}Jp(=UpfdSzkKx313kXzIv&!py5KqAWFk~!lWe;W{x9OnE|P#>A!_!>TKc!p~MI>Lj2v*AJx4gdfEi`%*GbYc?I00EB`&QAaU>N)3WvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
