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
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]],
            {self.images: images, self.labels: labels, self.is_training: False})
        return accuracy
       
    def predict(self, dataset, images, labels):
        predictions = self.session.run(self.predictions,
            {self.images: images, self.labels: labels, self.is_training: False})
        return predictions
       

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

        accuracy = network.evaluate(\"dev\", mnist.validation.images, mnist.validation.labels)
        print(\"{:.2f}\".format(100 * accuracy))

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    predictions = network.predict(\"test\", mnist.test.images, mnist.test.labels)
    test_labels = predictions

    for label in test_labels:
        print(label)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;7%VCR9yfW0)oax6*GjyvYTC`s}Z?E4>dgkac{%n=7fv}Z9C2yLvSj;GvI6~#Qz^B=K$n%0+Z_d+C_KFzILwpcdH>kw|Ezv+<XwK;+u~kCIG8b*anJS!qYZ}GW=YDQj@#}md3BWGf%*tbs;|bI=GjWo98HwhN7~}kJjB{<Rzv?<Vpw%MHO*J*QxzWYy<X^((dmK|L7EEowmLX#hse}m792jYV1(765r@m<Q!LGTIMHjrGe6Kn0YIJio3*BF|x58+pE#s@R+wnM2N`-*5)JXIyrm>d=}cJ^7EBjZ42$#s(^siEn9)*Z9r6!IksVt%%KNtqsStJB)f}+2AMtkD#><N+1MLk9(`-X+zfCW(H3#az)JRw*B;f`wBe9789Ww`tJr-=?Rlxsct(3e*YbH=AeGwt8?VJytJdS>UEhh$pBzWR+)wM7%l?!|TU!fkRd@%|NhD#1RE~#g6=<94*d{eo#kKHg-w(we`KLE>#j1tryfdJgG6WwenLeiECEA(+!^<2bD4REyN~Fxui-sqR`pFA~uu}REse{v=S?!bNU!a@6v~49rDkQNF7V()$^2Nl&P=1>=5%(f7sOTiCQzu7-V@>FgN$3Z{tOD?_t0k3syQLi7LP%Y3GsXD1#ZO+Ha>XA4If3<L-^=^B=iZx%*-Lwx@>dZYL?{0n_cr#@qYO}zp-i1j>oIX>g*V%a$po;$Ud<bJSPPZcg0ITGg7m&5fZB7BIgB-Vt$UUjZf3NS6zU>}9A@;)2A?AVBb_hc`)jlA`xIcM%=LSv5fVQ<yBQxlw9}Gz8ln!B19apV+ybD0%e+e@TB+o2%MI-I4=FVdd)>4T3jv+N>!)(KQJpDxMM|w;p4|{Pje%TH)BbnXe04i$<tW=_dEAYqC11*#sz)q419Ym>C5DH5KOJmO1eLzXk~GrA<wAyJ5H~~p$R@>g-%#Jodvy-rLw<|_yIchIXM}-{y_z(0zY?zk$2D57OHnO4&wE3AMw|KhK(kqm-(L_o^jn1r$t2-53#I!4%U9ATbZ;j;XWd}5J(HySTMr)?JnPdjg~4(jd-M)r;Rj~C58`vuN!t(8UX-KKm15c}Ig4dDqxxO0c~l1jnW!&aKAeZMh`bjSge4k~jO<ksHe1sF-w3;n0Y)IfsDp&bBTa**RRFz1`}xNmQuy_vRv*sLkal!dxJbp$!N;oBg7+PC*aSP=va7fNy~t<nv3qdzfe69xD^SdXO}X5&ez$poks$Lln1QE`j~?F$cBV?#Pq{`tRMw__Z>~wJj+K)HZA-`r>(Ip6@e<KPpf+aQLO`*7jT}hg7{6`*Fo`Y9Mh&~nks$jMJmw#{ESxQ)d*+N;TY-K~bN<z<OqMbOAfGe7B<(4e5B(l$SB#rN!Sw}_sjB+n!DVIAvMdukjtFVNZVWvv2UaeKu0kVhmJ>!T9r=iLL2C$jM_$FfnIYLoPTKRdNdK=oEw4evFWHfoiv8PMtNzjRTjB2H7$(hQVG|n!MYh@BT$tM$mUQpE=>T)IVa4ToDC=P~D~Hf;@F(83r;Zh$G$a7FD*>#pyEc!GdK^Z6x+9;L3xl_Lx4DF{C!yGv0@B*<JxP<+=+`M|)bhd0s+QS@u}8?GxFl!#FATgH<Ex67-J1}HNq_zZ^`_bZs7fMfR|&%;vc&W6!=E(w^w(cOuFrD$#Gv1kC?=#pu#{l`r42TB21cJ3Fl9s(eZ`TIU@nDFH`y1Axv#KENg4)DadqXJl8<E2V}uldFE33?&zkTlbGwU=1IB~6wWh1z8XPVulcE|_)%()r_OC}&<%HW86*^+pGK6(pB`*6>CgW5t?@Xl0K^mjEh(+@qYvMTB-pgl=Fo`{CL0!iw8rz_VjudJxK3-^|RT>7OCBj%Iq+MHFpyb+X3-LMh)Yvfn(w#17z}`aGQjy6c$dsR~NO9|IAPC6dIQo02wq_as9Xani$R;htIiKXTh8)lq{Bk3u&Wq^XH&mPMX6`<Iv8f_0<U8ii{(qpt0_fxi?n2wdjnDt?i06=oFFrxI-qr?2IwLj%cGQZFM4L2MWnDGNfOm+0zZH~DBq_swIb$IsBkm^9o@Q*Os&WM{Lr5;4pCAe{jl3#?C5w_JZR#*EU!TciYqp~~n5b&%alRpY7NG>pP4!ghbQ3PD4bo8R17n%={YB-$=u$h~B7ziERs^*vW$X|)f;5JlfcT0LkXGV?ad2xls@>nw&8p4!*_@PBKY?&!InddaT<NuZSB2mo$w_pT+C>Y#ZG9U{k-0BRJ|ud$9b68PWV4T&NCh@RpWIb$#sxS#`mfsEo3(G7MkE?p*y|*sOHVf!+O9kMgu;w2FAo>2C(5lT4Q|d<LcRKP$bg=Eo13v~v}Nk7O!9yaKqoaN2k8w{8(#dNo>>Hs5+M8!un1R{0R(&uNxX^2l@R_L1%w-8P@oJqa#5-EBD`a8dhvl9yyk4qaQ?mx*aaVxXbq8K=Lm-bJpe&0!T)P;M9JJ7I))bk%w~VY#H>EvxuVhIp7=bsG_v11jD54Tp?TZ0TDEd1_ok{$pORYhijma#iefa<LgvZbOBR}VUSkttcE1hkg}Q;_$Q1G&{+ydIW`XleL+$SpdxXw1!5(^=q#f3G;xJiPW03|V#e&fs@efgfQ;To^#=#X};X713DtQ~RK+w;wZT9vHMW7D@YBzrz?#81gBuccRtQoA&8@ESCT@_1YWl;yVDX#Y**6&E`H@~vop{g6m$le@19mDJ1N{;njq#6~)iJyYF5?3@Cua6A&I3(2c#s4}^|5p!%sbzbi{Ivl%paS`WaBaIe5&sqY0VyB|B9{2|)zRW*fEjEDvKuyO{LHYaegeIFkz<%jv6v=G$CiepFOp#=2>8yQ3%12qR(R>sKvXTMBaG8`<xLbtCoCUhgKMMtbSw9}fZxJmQ!c)4lQE;JL&G3Rh~X;{$pubva0ECugRD;g($JBD&lS5U@O5)DP=YnWsNMUv@2I6k3aX%Z1P+4_E7m`*@}w%0f)6NaXARFgz^K6zxkTKvPAI#~>6|qE_WXfq_51%G`5*d?3}{h*Jje+ruZ%>b<Pq3Al$;zKUd$Usq1huwk}`|wC@oL~RjW_!4b{$D-W!1nAW9j|1HmNH+Ebz2D;R`PtuntEh<Ge`-T9MN<#&p95R<hKG$Pk-k`XlNEvb}2K?%LQ`g(H<45rYt1Y6tOUzm(T%>)n+<z#zCTrF)#(CZxg_oKc@N$x{Q`uS9DZc^Wg7(;-ckK5~3;Qr3)exoAZRE6$MzG_@f0GNkn7fF?Q9FK18`9FPJSZ!<<ihg;9!307gf~}16O)GA`XGfQ>(%!4ksOy)tvm{2G5-rKK6j#U6&J(!U%QmyktO4v%{`EDGixUfRmA8z!RVl#G?!JoL&+f9TH3~T0bk|9!Y}7BM?(1H#fw7uVETZqO4f9#L%w9QWXQ26w1osWuC+2-^=+V2Bt_og9Wg5MRf;h5Nuv}HescXys%UqQ5Q*Air7VI&U7v?{@xWux%$R8;;CiTm{pF4frJz5jo!0&gw{R?tC&JC!GINB%JtbSbq*E{U^HlY&RatAkmTqaJB0b~_aC}Bv6rDsQx8JPOqOSd1Gv92jB=c3`MNLNa(J+%T&xY1~>YgIKRRY$d!yR}N#B4j;OyaFfaM}bE2YgQ9x9~9{({|X5<T(_F8KU0Pb4N{1Nl9#^0e`H5#EQin3dLi9E7;d>!6@9Wa8Q+0_n&UZR2VLfmuOnkyLyQxP?-2MSzk?XjY>z)Dip$ISL3NFiZ+Cw*G>;xy+-P)a)w<G;&uV{3ahzi;juaopY;u7ZuxM%*q<ymaEB>FA-#RP%)d5#d5BpnqbJZ$yVI~V$1o?5S=g+Y?##=Z13Ht})G*=z{8vxnd(iGo4`kx;(+$9VUhvD0JG6+CY5zm??HljzVqV24vZ)o~0!4#<OLUwhDdKvD90+o1wU{{q%o(4SbY!d51z^^GKw*kwOxbt1I1J)NA%=Cvz`zKT(FGeEa3{F2I<Mo=Iy1<>M9tqt+(DbX&*%1|l$chLOz86lWN5}-NX0dRA>IXg-A&xK$yEejg1GqC-(dhl2tC=VDPTbxEz*{~d%~@a?pg($sczygZLAMRpVA^YR>~q{$f9kK5A`Faqarz)~ha_#41kS8bMA#0rVa?KESuSpm!?ksCpNQOOjOQ9mhXN{EATjRl;pHfCo-+0N10}VLMOFX?m&CaAH3!-sbAQQen4d~l1<8o%l-+gDo}YXE6(I)2l*NXnO00w`0&lpy!YuL}gS3Imd>CSdFK!LjZHY^~2RhIbozp4#dUMXFj~VU3HE^tFqjth4Yyo7K9&o$8uuV+(fyVmZW5!`abP|4bx=M`y@HRggR#b-eL-MYlUTV;TN^SHfn|~|w%>R+?;8+S(LMn(Lt{bZyluHreDqV7Z+KI(S4klUh2F>5gA}J%2Dd5)qZ6<x;i)QUal>H-7UIDZ+%(nJK`SD2nVCRx8wAkRNXb16ue4Bb?Jru-Smb4SReGbxzFw%{s;-Xm&_6+n+Rz|N$5CkHhHoe|{HDqc<2x+JO;|^P;B7}3dt?&B4X4%&jUz(%ko98nL8-fE2$+5#yAX!*Rl8-7ERM{F!9QJoxY);rs-pz-9D7G5ZfBrjE!Hv4)HYVLQuf#sqcp6zCE0>waihch9ydkPRvg#OzFwxrJXM*g3RS%b!TxYylNU}w^z{0z&C*^$7=NfZ<F*oZSFsTB<(lPCCwiRX1`8bfaW-EXyFTg{yn>is&0XuG!KPrNPQ?J|;5uz$rziFH!yjT5O^o(pe9pFf`&mouDIxoiO=G}Mqw$N>OB?HfuIX!IMZRP&So>O>ig$L89gPc*17`{fRLKsVd`ID9Z$yARMyTK#hdH68quob+sv=n5eFiw(D6qO3JX-v*px*}E7u$yO8lkE$WJXn7I^STVdtfD+y+qYTwE9Qf_N}*E31n0v@#9|gfp=O@yl2(Zw&a|PSNQ3Zqn8P;MN4Y5s7@vPgu_a%`h~t>o0n$}d`tQ2QZIP|^z}r7>Q3s7)3==67K~o4PP83u8dEPq0VQF<T*t-=-t64dJ)HadN>lcY4`xe-%bxh4iK}#=IlV35k>kg3!zgbos4iz#Il2>F;_immrzg~wn(3%KE3KOE`2h9-V?nYF1AUY#Wv9?{_ou+=fWA>`+9^znnU$HeAIR@{QQ-i146f!bV5(4JI%n^v`SY6OV1EIb~SGrXyzK|87>Lh*CCrwGVd^&jB=VLd?56f|&8^F=Y!}tKXI+EuxuYxdHu9jO#g9k|UZwGb3y#dXvl4K%c{=r&FK5j<eu|b^=FXdA>AahyKd#I53O!EU4j~vrQn#umj!p1<nd3*jJO?`*F)N=CTXOK^BBw_8G-S>~fvzCXeEfV()c@<o#l@Rc5QpP{~L}uOpLCmTnV}v6l6}2x!`&~m8%T<Cie+&RQ^)6n(6(OO+WO?x=ZzF)W^1>DxmZb~V&0UHM{KVU){~gXkpxlA;q`h7C`4e~cD;Qwx+nV_)Rg<%{onz(*U--W&zn&{{lmd=;=sa^0_yET*6=RiZBES_WaduY3Xh0kq2>=lOYvb?)_S<KYyIvI9qf3mfu<J(jGbq$4VK70EnKtesSXHyovqTs;wpuxlAuFzT%j6A}JWY$?>t-(baS`1oUxt%IQjBBPu*{w9+b1N37~%?cOsT&Pzknj$rY=WkJF8<(zu&@i6fVPtG(4?klrNdcW6PLzD#>a;0d~e-mSsQ0_OzWiUoQ|btm6GjA?OZi84ijKM@twIJD}5W^5$=K2rwL7`W!K5b)~RBL8kH){->QA+QE9sG$Dczm}3G|3_Yo@P&et)d|vW~6kCiTMNdBdyw}5$Rn1;yi#NM1Y}y7X4cO3J%?J8EJ9B_PWCViQM#8=7cO`I#sQ1*5Q-<J*^GJF0@r~_(_8w1!YnNov?)RE|JO>1acQ*)1ac0q_0ZQkb@&u|%%j<Eor`8q2qR}DinZeAUpSC@wgFi?UuPMR8F@1>ZEQlx&${ET}(?oQUCBly8*dW=BbPXx?LSNaN5;_^Me1-`RJ0j|bFwX0MBGlk$$Ut0n##~dMawiobc3PlS__cB<{<!EAGV$;r0;EcBeID<>YE%S~&Sp5NFGK*KA{#*DH-N}E)`JUS*V?9zyi1l4v{uBy2fS*Gxr{O%;t3QYm=A~j!HgxWP=Py8>S76svD}rFTmZACyryYqIWv3gDY+Rx-#=+W!3UERaL3QDJVvc%v6Z!25SMZ<s$2(A>oGCQ^TK)iio#_<qDp%Wu%AgOQaTg{j1=JkBap8vnkw5XL6UN<tC2fR&lL)R4{t44CFM=T#2g-gof=8wJ;_VD*`ir>2o?YUPBC|~y)`9$00HnOpqv2!uzo){vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
