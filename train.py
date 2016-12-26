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
NumberOfStocks = 5
f_train = 0.8

####################
naturalVariation = None


# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):

    def __init__(self, n_stocks, n_hidden, train=True):
        super(RNNForLM, self).__init__(
            lb =L.Linear(n_stocks  , n_stocks),
            ll0=L.LSTM  (n_stocks  , n_stocks),
            ls1=L.Linear(n_stocks  , n_hidden),
            ls2=L.LSTM  (n_hidden  , n_hidden),
            ls3=L.LSTM  (n_hidden  , n_hidden),
            ls4=L.Linear(n_hidden  , n_stocks),

            lf =L.Linear(n_stocks, n_stocks),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.train = train

    def reset_state(self):
        self.ll0.reset_state()
        self.ls2.reset_state()
        self.ls3.reset_state()

    def __call__(self, x):
        h1 = self.lb (x)
        h2 = self.ll0(x)

        h  = self.ls1(x)
        h  = self.ls2(h)
        h  = self.ls3(h)
        h3 = self.ls4(h)

        h  = self.lf (h1+h2+h3)
        #h = self.lb (x)

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
            #print(x)
            #print(t)
            #print()
            #print(self.converter(batch, self.device).shape)
            #import pdb; pdb.set_trace()
            #x, t = concat_examples(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

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
    oridata = data[0:-1]
    newdata = data[1:]/oridata-1.
    data = newdata
    print(data)
    global naturalVariation
    naturalVariation = np.var(data)
    print("naturalVariation={0:e}".format(naturalVariation))
    n_data = len(data)
    data_train = data[:int(n_data*f_train) ].astype(np.float32)
    data_test  = data[ int(n_data*f_train):].astype(np.float32)
    ori_train  = oridata[:int(n_data*f_train) ].astype(np.float32)
    print("data loaded: #train={0:d}, #test={1:d}".format(len(data_train),len(data_test)))

    train_iter = ParallelSequentialIterator(data_train, args.batchsize, repeat=True )
    eval_iter  = ParallelSequentialIterator(data_train, args.batchsize, repeat=False)
    test_iter  = ParallelSequentialIterator(data_test , args.batchsize, repeat=False)

    # Prepare an RNNLM model
    rnn = RNNForLM(NumberOfStocks,int(NumberOfStocks/5))
    #model = L.Classifier(rnn,lossfun=chainer.functions.mean_squared_error)
    def my_mean_squared_error(x,t):
        #print(x.shape,t.shape)
        a= chainer.functions.mean_squared_error(x,t)
        #print(a.data)
        return a
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
    trainer.extend(extensions.snapshot(),trigger=(20,"epoch"))
    #trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}',trigger=(20,"epoch")))
    trainer.extend(extensions.ExponentialShift("alpha",0.5), trigger=(50, "epoch"))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    if args.test:
        chainer.serializers.load_npz(args.test, trainer)
        test_model = trainer.updater._optimizers["main"].target
        i = 0

        with open("output_train.csv","wa") as f:
            writer = csv.writer(f,lineterminator='\n')
        
            for x,t,ori in zip(data_train[:-1],data_train[1:],ori_train[1:]):
                if i%100==0:print(i)
                i+=1
                x = cuda.to_gpu(x.reshape(1,x.shape[0]))
                y = test_model.predictor(x)
                z = ori

                #writer.writerow([i]+list((y.data)[0])+list(t)+list(z))
                writer.writerow([i]+list((y.data)[0])+list(t)+list(z))
        return

    trainer.run()

if __name__ == '__main__':
    main()
