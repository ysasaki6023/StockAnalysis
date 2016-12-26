#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import cuda
import csv

####################
## Settings
NumberOfStocks = 50
f_train = 0.8

####################
naturalVariation = None

import numpy

from chainer import function
from chainer.utils import type_check


class myMeanSquaredError(function.Function):

    def __init__(self,x,multipl):
        self.x = x
        self.multipl = multipl
        super(myMeanSquaredError, self).__init__()

    def check_type_forward(self, in_types):
        #type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        y,t = inputs
        x = self.x
        self.diff = y - t
        diff = self.diff.ravel()
        wgt  = t/x - 1.
        wgt = wgt.ravel()
        self.wgt = wgt.dtype.type(1.) +  wgt.dtype.type(self.multipl)*wgt.dot(wgt)
        return numpy.array( self.wgt * diff.dot(diff) / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        y,t = inputs
        x = self.x
        self.diff = y - t
        diff = self.diff.ravel()
        wgt  = t/x - 1.
        wgt  = wgt.ravel()
        self.wgt = wgt.dtype.type(1.) +  wgt.dtype.type(self.multipl)*wgt.dot(wgt)
        return self.wgt * diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, inputs, gy):
        coeff = gy[0] * gy[0].dtype.type(2. / self.diff.size)
        gx0 = coeff * self.diff * self.wgt
        return gx0, -gx0

def my_mean_squared_error(x0, x1):
    t = x1[0]
    y = x0
    x = x1[1].data
    multipl = 1e6
    return myMeanSquaredError(x,multipl)(y, t)

# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):

    def __init__(self, n_stocks, n_hidden, n_memory, train=True):
        super(RNNForLM, self).__init__(
            lb0=L.Linear(n_stocks  , n_stocks, initialW=np.identity(n_stocks), initial_bias=0.),
            #lb0=L.Linear(n_stocks  , n_stocks),
            kkk=L.StatefulGRU(n_stocks  , n_stocks),

            #lb1=L.Linear(n_stocks  , n_hidden),
            #lb2=L.Linear(n_hidden  , n_hidden),
            #lb3=L.Linear(n_hidden  , n_hidden),
            #lb4=L.Linear(n_hidden  , n_stocks),
            #lb5=L.Linear(n_stocks  , n_stocks),


            #ll0=L.Linear(n_stocks  , n_memory),
            #ll1=L.StatefulGRU  (n_memory  , n_memory),
            #ll2=L.StatefulGRU  (n_memory  , n_stocks),
            #ll3=L.Linear(n_stocks  , n_stocks),
            #ll4=L.StatefulGRU  (n_stocks  , n_memory),
            #ll5=L.StatefulGRU  (n_memory  , n_stocks),
            #ll6=L.Linear(n_stocks  , n_stocks),

            #ls1=L.Linear(n_stocks  , n_hidden),
            #ls2=L.StatefulGRU  (n_hidden  , n_hidden),
            #ls3=L.StatefulGRU  (n_hidden  , n_hidden),
            #ls4=L.StatefulGRU  (n_hidden  , n_hidden),
            #ls5=L.StatefulGRU  (n_hidden  , n_hidden),
            #ls6=L.StatefulGRU  (n_hidden  , n_hidden),
            #ls7=L.StatefulGRU  (n_hidden  , n_hidden),
            #ls8=L.Linear(n_hidden  , n_stocks)

        )
        """
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        """

        #self.lb0.unchain()

        self.train = train

    def reset_state(self):
        """
        self.ll0.reset_state()
        self.ll1.reset_state()
        self.ll2.reset_state()
        self.ll4.reset_state()
        self.ll5.reset_state()
        self.ls2.reset_state()
        self.ls3.reset_state()
        self.ls4.reset_state()
        self.ls5.reset_state()
        self.ls6.reset_state()
        self.ls7.reset_state()
        """

    def __call__(self, x):
        #self.lb0.W.unchain_backward()
        #self.lb0.b.unchain_backward()

        h0 = self.lb0(x)
        h1 = self.kkk(x)

        """
        h  = F.dropout(F.relu(self.lb1(x)),train=self.train)
        h  = F.dropout(F.relu(self.lb2(h)),train=self.train)
        h  = F.dropout(F.relu(self.lb3(h)),train=self.train)
        h  = F.dropout(F.relu(self.lb4(h)),train=self.train)
        h1 = F.dropout(F.relu(self.lb5(h)),train=self.train)

        h  = F.dropout(F.relu(self.ll0(x)),train=self.train)
        h  = F.dropout(F.relu(self.ll1(h)),train=self.train)
        h  = F.dropout(F.relu(self.ll2(h)),train=self.train)
        h  = F.dropout(F.relu(self.ll3(h)),train=self.train)
        h  = F.dropout(F.relu(self.ll4(h)),train=self.train)
        h  = F.dropout(F.relu(self.ll5(h)),train=self.train)
        h2 = F.dropout(F.relu(self.ll6(h)),train=self.train)

        h  = F.relu(self.ls1(x))
        h  = F.dropout(F.relu(self.ls2(h)),train=self.train)
        h  = F.dropout(F.relu(self.ls3(h)),train=self.train)
        h  = F.dropout(F.relu(self.ls4(h)),train=self.train)
        h  = F.dropout(F.relu(self.ls5(h)),train=self.train)
        h  = F.dropout(F.relu(self.ls6(h)),train=self.train)
        h  = F.dropout(F.relu(self.ls7(h)),train=self.train)
        h3 = F.relu(self.ls8(h))

        h  = h0+h1+h2+h3
        """
        h  = h0+h1

        return h

# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)

# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_RMS(result):
    #print(result)
    result['RMS'] = np.sqrt(result['main/loss'])
    result['RMS/nVar'] = np.sqrt(result['main/loss'])/naturalVariation


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)
            #print(self.converter(batch, self.device).shape)
            #import pdb; pdb.set_trace()
            #x, t = concat_examples(batch, self.device)

            # Compute the loss at this time step and accumulate it
            # Trick to pass x
            loss += optimizer.target(chainer.Variable(x), (chainer.Variable(t),chainer.Variable(x)))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters
        #print(dir(self))
        #raw_input()

class myClassifier(L.Classifier):
    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        accur = 0.
        print(self.y)
        row_input("pose")
        reporter.report({'Accuracy': accur}, self)
        return self.loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=6,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=1000000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=6, # past 30min
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', '-t', default=None,
                        help='Test using the learnt model from snapshot')
    parser.add_argument('--analysis', '-a', default=None,
                        help='Analize the learnt model from snapshot')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()

    # Load file
    #import h5py
    #infile = h5py.File("all.hdf5","r")
    #print infile["data"].value
    #column = np.loadtxt("test.csv",delimiter=",",skiprows=0,usecols=range(1,NumberOfStocks+1))
    data   = np.loadtxt("test.csv",delimiter=",",skiprows=1,usecols=range(1,NumberOfStocks+1))
    #oridata = data[0:-1]
    newdata = data
    data = newdata
    print(data)
    global naturalVariation
    naturalVariation = np.var(data[1:]-data[:-1])
    print("naturalVariation={0:e}".format(naturalVariation))
    n_data = len(data)
    data_train = data[:int(n_data*f_train) ].astype(np.float32)
    data_test  = data[ int(n_data*f_train):].astype(np.float32)
    print("data loaded: #train={0:d}, #test={1:d}".format(len(data_train),len(data_test)))

    train_iter = ParallelSequentialIterator(data_train, args.batchsize, repeat=True )
    eval_iter  = ParallelSequentialIterator(data_train, args.batchsize, repeat=False)
    test_iter  = ParallelSequentialIterator(data_test , args.batchsize, repeat=False)

    # Prepare an RNNLM model
    rnn = RNNForLM(NumberOfStocks,int(NumberOfStocks/5),int(NumberOfStocks*50))
    #model = L.Classifier(rnn,lossfun=chainer.functions.mean_squared_error)
    model = L.Classifier(rnn,lossfun=my_mean_squared_error)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    #optimizer = chainer.optimizers.SGD(lr=1.0)
    #optimizer = chainer.optimizers.Adam(alpha=1e-10)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    eval_rnn.train = False

    #trainer.extend(extensions.Evaluator(test_iter, eval_model, device=args.gpu,eval_hook=lambda _: eval_rnn.reset_state()))
    #trainer.extend(extensions.Evaluator(train_iter, eval_rnn, device=args.gpu,eval_hook=lambda _: eval_rnn.reset_state(),eval_func=myEval))
    #trainer.extend(extensions.Evaluator(eval_iter, eval_rnn, device=args.gpu,eval_hook=lambda _: eval_rnn.reset_state(),eval_func=myEval))
    #trainer.extend(extensions.Evaluator(test_iter, eval_rnn, device=args.gpu,eval_hook=lambda _: eval_rnn.reset_state(),eval_func=myEval))

    interval = 10
    trainer.extend(extensions.LogReport(postprocess=compute_RMS,trigger=(interval, 'iteration'),log_name="log.dat"))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'RMS','RMS/nVar']), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=1 if args.test else 10))
    trainer.extend(extensions.snapshot(),trigger=(1,"epoch"))
    #trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}',trigger=(20,"epoch")))
    trainer.extend(extensions.ExponentialShift("alpha",0.5), trigger=(20, "epoch"))
    if args.analysis:
        import h5py
        chainer.serializers.load_npz(args.analysis, trainer)
        test_model = trainer.updater._optimizers["main"].target.predictor
        with h5py.File(args.analysis+".hdf5","w") as f:
            test_model.to_cpu()

            f.create_dataset("lin.W.W",data=test_model.lb0.W.data)
            f.create_dataset("lin.W.b",data=test_model.lb0.b.data)
            f.create_dataset("GRU.W.W",data=test_model.kkk.W.W.data)
            f.create_dataset("GRU.W.b",data=test_model.kkk.W.b.data)
            f.create_dataset("GRU.U.W",data=test_model.kkk.U.W.data)
            f.create_dataset("GRU.U.b",data=test_model.kkk.U.b.data)
            f.create_dataset("GRU.U_r.W",data=test_model.kkk.U_r.W.data)
            f.create_dataset("GRU.U_r.b",data=test_model.kkk.U_r.b.data)
            f.create_dataset("GRU.U_z.W",data=test_model.kkk.U_z.W.data)
            f.create_dataset("GRU.U_z.b",data=test_model.kkk.U_z.b.data)
            f.create_dataset("GRU.W_r.W",data=test_model.kkk.W_r.W.data)
            f.create_dataset("GRU.W_r.b",data=test_model.kkk.W_r.b.data)
            f.create_dataset("GRU.W_z.W",data=test_model.kkk.W_z.W.data)
            f.create_dataset("GRU.W_z.b",data=test_model.kkk.W_z.b.data)
        return

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if args.test:
        chainer.serializers.load_npz(args.test, trainer)
        test_model = trainer.updater._optimizers["main"].target
        i = 0

        with open("output_train.csv","wa") as f:
            writer = csv.writer(f,lineterminator='\n')
        
            for x,t in zip(data_train[:-1],data_train[1:]):
                if i%100==0:print(i)
                i+=1
                x = cuda.to_gpu(x.reshape(1,x.shape[0]))
                y = test_model.predictor(x)

                writer.writerow([i]+list((y.data)[0])+list(t))
        return

    trainer.run()

if __name__ == '__main__':
    main()
