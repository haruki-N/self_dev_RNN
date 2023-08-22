import sys
import os
sys.path.append(os.path.abspath())
import numpy as np
import argparse
from optimizer import SGD
from data_ptb import *
from rnn_lm import *



def get_args():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--wordvec_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--time_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()
    return args


def train():
    args = get_args()

    # load dataset
    corpus, word_to_id, id_to_word = load_data('train')
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    xs = corpus[:-1]   # train input
    ts = corpus[1:]   # train gold
    data_size = len(xs)
    print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

    # prepare for training
    max_iters = data_size // (args.batch_size * args.time_size)
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ppl_list = []

    # prepare for model
    model = SimpleRnnLM(vocab_size, args.wordvec_size, args.hidden_size)
    optimizer = SGD(lr=args.lr)

    # train
    jump = (corpus_size - 1) // args.batch_size
    offsets = [i * jump for i in range(args.batch_size)]

    for epoch in range(args.epoch):
        # ミニバッチの取得
        batch_x = np.empty((args.batch_size, args.time_size), dtype='i')
        batch_t = np.empty((args.batch_size, args.time_size), dtype='i')
        for t in range(args.time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

            # 勾配を求め、パラメータを更新
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1


        ppl = np.exp(total_loss / loss_count)
        print('| epoch %d | perplexity %.2f' % (epoch+1, ppl))
        ppl_list.append(float(ppl))
        total_loss, loss_count = 0, 0





