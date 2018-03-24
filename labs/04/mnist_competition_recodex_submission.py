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

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;3p3XBV7O*0)vfMVRLir<IYcKWFrJ+4qv*S3u^lCyPR^80EVLmXb)(-K56;R2h-4g{zR{kGo#nxuc#?I(A{l-f3mJ--7C1^*G{ThQR7?u`FIC<2$s~Xyz0FQ-yWJjV4YcHd;7mFcg^(uM1!ghc2VyiwecduWkS4xT*0aNx>$d|YNz1Hg38Etq;Dyi#A$wH8a3$6cGykLfbv<lc%t-B9a@;EOVBWcQrbd$8xV?SJ`<^$9o#QXJ!v8MsKg+j*OJ9gbT*#zH(6`ADUUQ6jo1^cPYOP%ARXx833<X@`^INNQL&>}yA?QfarFb`Sw)Q-pJ>sKYgF>;ZJG&VM3i|WGFiZUBIDSxewFokwYq*=`V}w;X-W;-OGDiqg6KE0Y*PNh=58`#f`CT$&|%r+6-MJeD;hN+WdTRl*i74t5HWiw%|Q-#ES^B`Tu_v@2OAyY2uzxtiy9-QpAiEYo+7Mj`qXD?>s12Nh3F47S_1Cz4JhT{hbg{UhOV^C%~>_I3;8kIVLb8m|5C2F>ES904jE%&MIf~S7@`q>(H25%D4!8|BDQE4qa6ihtgEr%F%Nz>73q*3K!$4M!yIC|p5$Dqj)dFcpY|q}lDt&DscrNirZXZlSdIuN359L+oqFjpyr??fvh=m;vrCl~TA>nv7;y4NP7$^C4lj3f(Wv4hwy#3N?3EbS>G(D}4MiJusKT}J1n$VScp2MBxx73Hj}gS2(sNKb)x20_l}1&xI8(Sfb#nnko^?7w$_6z}rsnnGn=1r6kpNm|E~lEHd)Eln!6b`3S1e@Yog=ZBwoe@Q@MvPjh#I8mzzSFRx6Sj24XE_HNd;j>5PzO4v@+);Kd-<mLxLjHJ6&s9yASqY7I}fekKh_M)kQG%lB$yjM?>L0qGj14iuvej>egB+?DwR)t$=FXO;!<R@R>{qAzH5G3}za5Dw{JAL_sw(IOLwfm<oL|(Eq{+Wu)Z!+p4?edTKgCDBm43sO|n_U8ZGKtPqrt;lw1yKasP2>UX>~koGa)U6%I7LiS}ffLDIX&a$4&Nc|BuV)xod>{F3pLGKB?6-ZUH2eMYc5gasYsS4VG)6pT6z>5THXv^0Lo-=J-as63RT~Y{OCPgFRgty|I4zk6*!IeWE;L&>gTMmy5hj|1<ereZ3Bv&dyUH8I>bvH<kXAAjlvQ5%O=XF(oU(v*Z4BN<7Ga@y-XvB)@$K^~Vf^Jk%cxK?j@*xi@gf$xL<|14i^O8aX(M6s&)p@9YL3_7tsK*27QN&*zUnxQwzozMB)c7s1(XSz}(c~upx4bpIxj`5R$eG1UsA3k-X<uv7DIW9;w!<1w80yx3S(LV823)Lc{$dSe)HuYm4GE=QcmjS)(UVbt`E!t%L(;F+=&xzgMt~tUvX<%)`$*=fP_vfcVC1b>vVlb(smGv9kh!;#ra33_!BvkJ`OJ5j+%oEag)$;t2!g9wJOcAI+}}7;1Yqh?tJE^zpff<~DErDK@S5V|#WBRaJNMmTKQ@-KZ-aB6q<9bW^2viroJdEXBwr^#6)p7b`p&cos6YZX2aerZU+0<G*y#H~P^ukxK`IO{+pnpGAS_Br@LuOcuz=)fd&k;F<a9{@)-lkp+GhQEUdY+G70bSV3nNj0WaYW;U7hz^H2beVtZ4$@?5y2VoPlEPC;LsqEd*v0!AkfY9`XD5)#t~$(iQrT^D$96ha%=;g`z(3tbsfSR#d0V?PaB}=QtWd(;ae09{B<RhC``>(0d%G^j%G_(my9KKsP{3(!=)W%7-@hh#EuJby9iBRz-SXIa+Zl^G=r-l{JqI<}m51>cf5(ibQ9sVr#%I>hPrv>nNrTH4G3^IdwUKHYrwy(5#<y$f)m*lhX)BNP=pAEGqON3h&<rK$4fF210QYIpe%HV(v<!8SrWB;z3UDnQVX9X9V8rx^MCd*$s4;0CR4xcz_j@7io&jA4J(E@l;dgMQt_$AnYAi>V!$Fg_%dJo_qA5zY4nFP%z+zA|=7s1JMyAhI32%aF^>t@v$^shfkN?&SKjr7Yg+@vQxIKmK=%*zA#pSNGrTS93nbZk<k}H^tU=H3!h&QoxrPZ9q;`LlDG4H%zj;cHUT*_zE@1(fq0$~La_TgI!EfEwwU+PqvD`Yrj8P!?Q*Wh4C(or*SL-<&JytF$Q!sD)lIAJcnE5z%>>DzUf`?#9j+>UMYO)$<Yud}OEv+Nzmr#Lz##MHPDaKT>k$ZD_#I_5O4RnNX9B8rtJ@`VN{_^Q6ocmPa)L)HU?J(CyTt(P60+Zeq>qdj{KccM$QRJD)+`La_)!K;uizeq=0)Rgd|i9xCbkT9NA|y9$G_GP0Wvx7ZS^rhL+i(yy%v<s^~?)`cyT%?sPTkIVh_tjM8`){uGA53RSOpBcI2xH8$V3@&Svs_Xkkb_vBe#J9Gw4Xs`-zH=o=si`&XT72y6{$>Q$5V*5z}q<C8znMli`(>45QU!DdQa*PV`a&IgWjcox_d5r6BYD%4f_R3yO&g0HaiSJH73{50k;pMX=M?>TG@yYe=?2%@Rw3JA_9j4aqo0-?d(itt|7aOin3<1Fq%R^<FBX-;O|()egzc?8E;Ia7CkNh<gKKiJc>r4+!ua;@@<*(ABb;evJfHCfYyBO<wKZ~oimyvE(YWSwB^Tgpv+R`TjB(1dF+U%80tm~_!Xk4n1sve6Qyke{Sda^H7ZeZpVKr%+=T_R|!S2I<iLwdfx5BHXS7r>r4TmleqIBvzwe2^6vIq*n>NNpu091sp?6>i4JX{RpU8pnm~>E8;y`g%iwqj;kQ<TC7d@R0IAuB#B=3Jfm`0Ko!$(#fk+b3<P1aG3D>Oh@IO!#p67>n*PEJLG*dk0B?hRWK;L39i<6|++d(T>b8a+?fFSF&k#!0Nl9q{V;!)J8E|^A+x+6<A<7)Lws>IeKTWsnqj9Er{PqPyhfF{cqiw+gNJoRgrooFvN{1+uxH{K~xILyk^|guhX~(R=MWUdsY^Dy{TJbD5HWy-&Wwu5_<G1Q_!3J1Vsk1{9(@CmGQF6F#rAaz7G~|BIVrkXk8ixy7vy#;L?rWp(CmiSv2lrKyO)??Oj0sg#<>fxde`KF)=G^4zM-5GR8hMQ2K`0fVXr~#z<um!&v8*=8ZM`|64t3z%=<fo=A5Ho<y9BL*VqN;b16snD1F*?ty8BFBOxheffM7rTYO%}zakqbcrw>&>fZuk9;J4X#(02O5uU?(wl5d*Hc38%x#s)0?8BMJGFkFK)T65c~SVBacr&tRTcK?&2UG4R!w_>`JW10PE%#3e5`6)1e#vR|WDw6UWQCv*mF5pwdSuID~9;=n&t3m((by&EQ?eo^_00F-gkWK&qQ3dUKvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
