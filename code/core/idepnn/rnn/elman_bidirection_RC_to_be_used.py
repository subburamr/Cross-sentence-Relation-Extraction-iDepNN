import theano
import theano.tensor as T
import numpy as np
import pickle as pickle
import os
import datetime
from collections import OrderedDict
import datetime
from theano import config
from utils.features import numpy_floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from optimiser.grad_optimiser import create_optimization_updates
# from recursive_net_utils.tree_rnn import TreeRNN

SEED = 123
np.random.seed(SEED)

class EB_RNN_4(object):
    def __init__(self, nin, n_hidden=100, nout=2, learning_rate_decay=1.0, activation='tanh', optimiser='sgd',
                 output_type='softmax', L1_reg=0.00, L2_reg=0.00001, state=None, vocab_size=None,
                 dim_emb=50, context_win_size=5, embedding=None, use_dropout=False,
                 w_hh_initialise_strategy='identity', w_hh_type='independent', reload_model=None,
                 reload_path=None,
                 w2v_embedding=None, position_feat= False, entity_presence_feat=False,
                 pos_feat_embedding=False, pos_indicator_embedding=False, ent_pres_feat_embedding=False,
                 dim_ent_pres_emb = 50, dim_pos_emb = 50, pos_vocab_size = None,
                 pos_emb_type=None, ent_pres_emb_type=None, context_window_usage=False,
                 postag = False, postag_vocab_size = None, entity_class = False,
                 entity_class_vocab_size = None, dim_postag_emb = 5, dim_entity_class_emb = 5,
                 update_pos_emb = None, update_ner_emb = None, add_subtree_emb = False,
                 dim_subtree_emb = None, max_degree=None, treernn_weights = 'independent',
                 margin_pos=2.5, margin_neg=0.5, scale=2,
                 batch_size=1, ignore_class=18, ranking=False):

        rng = np.random.RandomState(1234)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activ = activation
        self.output_type = output_type
        self.use_dropout = use_dropout
        self.optimiser = optimiser
        self.w_hh_type = w_hh_type
        self.position_feat = position_feat
        self.entity_presence_feat = entity_presence_feat
        self.pos_feat_embedding = pos_feat_embedding
        self.pos_indicator_embedding = pos_indicator_embedding
        self.ent_pres_feat_embedding = ent_pres_feat_embedding
        self.dim_ent_pres_emb = dim_ent_pres_emb
        self.dim_pos_emb = dim_pos_emb
        self.pos_vocab_size = pos_vocab_size
        self.pos_emb_type = pos_emb_type
        self.ent_pres_emb_type = ent_pres_emb_type
        self.postag = postag
        self.dim_postag_emb = dim_postag_emb
        self.postag_vocab_size = postag_vocab_size
        self.entity_class = entity_class
        self.dim_entity_class_emb = dim_entity_class_emb
        self.entity_class_vocab_size = entity_class_vocab_size
        self.update_pos_emb = update_pos_emb
        self.update_ner_emb = update_ner_emb
        self.add_subtree_emb = add_subtree_emb
        self.dim_subtree_emb = dim_subtree_emb
        self.num_emb = len(w2v_embedding)
        self.emb_dim = dim_subtree_emb
        self.hidden_dim = dim_subtree_emb
        self.max_degree = max_degree
        self.treernn_weights = treernn_weights

        if embedding == 'word2vec_update':
            # vocab_size : size of vocabulary
            # dim_emb : dimension of the word embeddings
            # context_win_size : word window context size
            self.vocab_size = vocab_size
            self.dim_emb = dim_emb
            self.context_win_size = context_win_size
            self.emb = theano.shared(name='embeddings',
                                     value=np.array(w2v_embedding).astype(theano.config.floatX))

            if entity_presence_feat == True:
                ent_pres_feat = T.fmatrix()# as many rows as words in sentence, columns are two

            if position_feat == True:
                pos_feat = T.fmatrix() # as many rows as words in sentence, columns are two

            if pos_feat_embedding == True:
                if self.pos_emb_type == 'COUPLED':
                    self.pos_emb_e1_e2 = theano.shared(name='pos_emb_e1_e2',
                                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                     (2, self.pos_vocab_size+1, self.dim_pos_emb)).astype(theano.config.floatX))
                    # add one for padding at the end

                elif self.pos_emb_type == 'DECOUPLED':
                    self.pos_emb_e1 = theano.shared(name='pos_emb_e1',
                                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                  (self.pos_vocab_size+1, self.dim_pos_emb)).astype(theano.config.floatX))
                    # add one for padding at the end
                    self.pos_emb_e2 = theano.shared(name='pos_emb_e2',
                                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                  (self.pos_vocab_size+1, self.dim_pos_emb)).astype(theano.config.floatX))
                    # add one for padding at the end
            if ent_pres_feat_embedding == True:
                if self.ent_pres_emb_type == 'COUPLED':
                    self.ent_pres_emb_e1_e2 = theano.shared(name='ent_pres_emb_e1_e2',
                                                            value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                          (2, self.vocab_size+1, self.dim_ent_pres_emb)).astype(theano.config.floatX))
                    # add one for padding at the end

                elif self.ent_pres_emb_type == 'DECOUPLED':
                    self.ent_pres_emb_e1 = theano.shared(name='ent_pres_emb_e1',
                                                         value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                       (self.vocab_size+1, self.dim_ent_pres_emb)).astype(theano.config.floatX))
                    # add one for padding at the end
                    self.ent_pres_emb_e2 = theano.shared(name='ent_pres_emb_e2',
                                                         value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                       (self.vocab_size+1, self.dim_ent_pres_emb)).astype(theano.config.floatX))
                    # add one for padding at the end



            if postag == True:
                self.postag_emb = theano.shared(name='postag_emb',
                                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                              ( self.postag_vocab_size, self.dim_postag_emb)).astype(theano.config.floatX))
                # add one for padding at the end

            if entity_class == True:
                self.entity_class_emb = theano.shared(name='entity_class_emb',
                                                      value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                    ( self.entity_class_vocab_size, self.dim_entity_class_emb)).astype(theano.config.floatX))
                # add one for padding at the end

        if embedding == 'theano_word_embeddings':
            # vocab_size : size of vocabulary
            # dim_emb : dimension of the word embeddings
            # context_win_size : word window context size
            self.vocab_size = vocab_size
            self.dim_emb = dim_emb
            self.context_win_size = context_win_size
            self.emb = theano.shared(name='embeddings',
                                     value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                   (self.vocab_size+1, self.dim_emb)).astype(theano.config.floatX))
            # add one for padding at the end
        #define shared variables
        # for embedding type word2vec_update, nin = self.dim_emb*self.context_win_size
        # else nin = vocab_size (for one-hot encoding) and nin = dim_emb+pos+entity_pres
        '''
        self.W_xh = np.asarray(
			    rng.normal(size=(nin, n_hidden), scale= .01, loc = .0), dtype = theano.config.floatX)
		'''
        self.W_xh_f = np.asarray(
            rng.normal(size=(nin, n_hidden), scale= .01, loc = .0), dtype = theano.config.floatX)
        self.W_xh_b = np.asarray(
            rng.normal(size=(nin, n_hidden), scale= .01, loc = .0), dtype = theano.config.floatX)

        if self.use_dropout == True:
            # Used for dropout
            use_noise = theano.shared(numpy_floatX(0.))
            mask = T.matrix('mask', dtype=config.floatX)
            trng = RandomStreams(SEED)

        max_norm = None
        if reload_model != True:
            if state['clipstyle'] == 'rescale':
                self.clip_norm_cutoff = state['cutoff']
                max_norm = self.clip_norm_cutoff

        if self.w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            if w_hh_initialise_strategy == 'identity':
                # to handle 'vanishing gradient' problem
                self.W_hh = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))
                self.W_hh_bi = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))
            elif w_hh_initialise_strategy == 'ortho':
                self.W_hh = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)
                self.W_hh_bi = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)
            elif w_hh_initialise_strategy == 'uniform':
                self.W_hh = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)
                self.W_hh_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)
            else:
                self.W_hh = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)
                self.W_hh_bi =  np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)
        else:
            if w_hh_initialise_strategy == 'identity':
                # to handle 'vanishing gradient' problem
                self.W_hh_f = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))
                self.W_hh_b = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))
                self.W_hh_bi = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))
            elif w_hh_initialise_strategy == 'ortho':
                self.W_hh_f = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)
                self.W_hh_b = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)
                self.W_hh_bi = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)
            elif w_hh_initialise_strategy == 'uniform':
                self.W_hh_f = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)
                self.W_hh_b = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)
                self.W_hh_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)
            else:
                self.W_hh_f = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)
                self.W_hh_b = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)
                self.W_hh_bi = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)

        self.W_hy_f = np.asarray(
            rng.normal(size=(n_hidden, nout), scale =.01, loc=0.0), dtype = theano.config.floatX)
        self.W_hy_b = np.asarray(
            rng.normal(size=(n_hidden, nout), scale =.01, loc=0.0), dtype = theano.config.floatX)
        self.W_hy_bi = np.asarray(
            rng.normal(size=(n_hidden, nout), scale =.01, loc=0.0), dtype = theano.config.floatX)

        self.b_hh_f = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.b_hh_b = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.b_hy_f = np.zeros((nout,), dtype=theano.config.floatX)
        self.b_hy_b = np.zeros((nout,), dtype=theano.config.floatX)
        self.b_hy_bi = np.zeros((nout,), dtype=theano.config.floatX)

        if self.activ == 'tanh':
            activation = T.tanh
        elif self.activ == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activ == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activ == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError

        self.activ = activation
        #define symbolic variables
        if embedding == 'theano_word_embeddings' or embedding == 'word2vec_update':
            idx_f = T.imatrix() # as many columns as context window size/lines as words in the sentence
            idx_b = T.imatrix() # as many columns as context window size/lines as words in the sentence

            if context_window_usage == True:
                x_f = self.emb[idx_f].reshape((idx_f.shape[0], self.dim_emb*self.context_win_size))
                x_b = self.emb[idx_b].reshape((idx_b.shape[0], self.dim_emb*self.context_win_size))
            else:
                x_f = self.emb[idx_f].reshape((idx_f.shape[1], self.dim_emb*self.context_win_size))
                x_b = self.emb[idx_b].reshape((idx_b.shape[1], self.dim_emb*self.context_win_size))

            if self.position_feat == True and self.entity_presence_feat == True:
                x_f = theano.tensor.concatenate([x_f, ent_pres_feat, pos_feat], axis=1)
                x_b = theano.tensor.concatenate([x_b, ent_pres_feat[::-1], pos_feat[::-1]], axis=1)


            elif self.entity_presence_feat == True:
                # concatenate entity presence features
                # concatenate entity presence features to embedding of x (list of word indices)
                x_f = theano.tensor.concatenate([x_f, ent_pres_feat], axis=1)
                x_b = theano.tensor.concatenate([x_b, ent_pres_feat[::-1]], axis=1)

            elif self.position_feat == True:
                # concatenate position features
                x_f = theano.tensor.concatenate([x_f, pos_feat], axis=1)
                x_b = theano.tensor.concatenate([x_b, pos_feat[::-1]], axis=1)

            if pos_feat_embedding == True:
                pos_feat_idx = T.imatrix()
                if self.pos_emb_type == 'COUPLED':
                    pos_feat_emb_f_e1 = self.pos_emb_e1_e2[0][pos_feat_idx[:,0]].reshape((pos_feat_idx[:,0].shape[0],
                                                                                          self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_f_e2 = self.pos_emb_e1_e2[1][pos_feat_idx[:,1]].reshape((pos_feat_idx[:,1].shape[0],
                                                                                          self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_b_e1 = self.pos_emb_e1_e2[0][pos_feat_idx[:,0][::-1]].reshape((pos_feat_idx[:,0].shape[0],
                                                                                                self.dim_pos_emb*self.context_win_size))

                    pos_feat_emb_b_e2 = self.pos_emb_e1_e2[1][pos_feat_idx[:,1][::-1]].reshape((pos_feat_idx[:,1].shape[0],
                                                                                                self.dim_pos_emb*self.context_win_size))
                elif self.pos_emb_type == 'DECOUPLED':
                    pos_feat_emb_f_e1 = self.pos_emb_e1[pos_feat_idx[:, 0]].reshape((pos_feat_idx[:, 0].shape[0],
                                                                                     self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_f_e2 = self.pos_emb_e2[pos_feat_idx[:, 1]].reshape((pos_feat_idx[:,1].shape[0],
                                                                                     self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_b_e1 = self.pos_emb_e1[pos_feat_idx[:, 0][::-1]].reshape((pos_feat_idx[:, 0].shape[0],
                                                                                           self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_b_e2 = self.pos_emb_e2[pos_feat_idx[:, 1][::-1]].reshape((pos_feat_idx[:,1].shape[0],
                                                                                           self.dim_pos_emb*self.context_win_size))
            if ent_pres_feat_embedding == True:
                ent_pres_feat_idx = T.imatrix()
                if self.pos_emb_type == 'COUPLED':
                    ent_pres_feat_emb_f_e1 = self.ent_pres_emb_e1_e2[0][ent_pres_feat_idx[:,0]].reshape((ent_pres_feat_idx[:,0].shape[0],
                                                                                                         self.dim_ent_pres_emb*self.context_win_size))

                    ent_pres_feat_emb_f_e2 = self.ent_pres_emb_e1_e2[1][ent_pres_feat_idx[:,1]].reshape((ent_pres_feat_idx[:,1].shape[0],
                                                                                                         self.dim_ent_pres_emb*self.context_win_size))

                    ent_pres_feat_emb_b_e1 = self.ent_pres_emb_e1_e2[0][ent_pres_feat_idx[:,0][::-1]].reshape((ent_pres_feat_idx[:,0].shape[0],
                                                                                                               self.dim_ent_pres_emb*self.context_win_size))

                    ent_pres_feat_emb_b_e2 = self.ent_pres_emb_e1_e2[1][ent_pres_feat_idx[:,1][::-1]].reshape((ent_pres_feat_idx[:,1].shape[0],
                                                                                                               self.dim_ent_pres_emb*self.context_win_size))

                elif self.pos_emb_type == 'DECOUPLED':
                    ent_pres_feat_emb_f_e1 = self.pos_emb_e1[ent_pres_feat_idx[:, 0]].reshape((ent_pres_feat_idx[:, 0].shape[0],
                                                                                               self.dim_ent_pres_emb*self.context_win_size))
                    ent_pres_feat_emb_f_e2 = self.pos_emb_e2[ent_pres_feat_idx[:, 1]].reshape((ent_pres_feat_idx[:,1].shape[0],
                                                                                               self.dim_ent_pres_emb*self.context_win_size))
                    ent_pres_feat_emb_b_e1 = self.pos_emb_e1[ent_pres_feat_idx[:, 0][::-1]].reshape((ent_pres_feat_idx[:, 0].shape[0],
                                                                                                     self.dim_ent_pres_emb*self.context_win_size))
                    ent_pres_feat_emb_b_e2 = self.pos_emb_e2[ent_pres_feat_idx[:, 1][::-1]].reshape((ent_pres_feat_idx[:,1].shape[0],
                                                                                                     self.dim_ent_pres_emb*self.context_win_size))

            if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                x_f = theano.tensor.concatenate([x_f, ent_pres_feat_emb_f_e1, ent_pres_feat_emb_f_e2,
                                                 pos_feat_emb_f_e1, pos_feat_emb_f_e2], axis=1)
                x_b = theano.tensor.concatenate([x_b, ent_pres_feat_emb_b_e1, ent_pres_feat_emb_b_e2,
                                                 pos_feat_emb_b_e1, pos_feat_emb_b_e2], axis=1)
            elif self.ent_pres_feat_embedding == True:
                x_f = theano.tensor.concatenate([x_f, ent_pres_feat_emb_f_e1, ent_pres_feat_emb_f_e2], axis=1)
                x_b = theano.tensor.concatenate([x_b, ent_pres_feat_emb_b_e1, ent_pres_feat_emb_b_e2], axis=1)

            elif self.pos_feat_embedding == True:
                x_f = theano.tensor.concatenate([x_f, pos_feat_emb_f_e1, pos_feat_emb_f_e2], axis=1)
                x_b = theano.tensor.concatenate([x_b, pos_feat_emb_b_e1, pos_feat_emb_b_e2], axis=1)

            if self.postag == True:
                # postag_idx = T.imatrix()
                postag_idx = T.ivector()

                # self.postag_emb_val = self.postag_emb[postag_idx].reshape((postag_idx.shape[0],
                                                        # self.dim_postag_emb*self.context_win_size))
                # self.postag_emb_val = self.postag_emb[postag_idx[:, 0]].reshape((postag_idx.shape[1], self.dim_postag_emb))
                self.postag_emb_val = self.postag_emb[postag_idx]
                x_f = theano.tensor.concatenate([x_f, self.postag_emb_val], axis=1)
                x_b = theano.tensor.concatenate([x_b, self.postag_emb_val[::-1]], axis=1)

            if self.entity_class == True:
                entity_class_idx = T.ivector()
                self.entity_class_emb_val = self.entity_class_emb[entity_class_idx]
                x_f = theano.tensor.concatenate([x_f, self.entity_class_emb_val], axis=1)
                x_b = theano.tensor.concatenate([x_b, self.entity_class_emb_val[::-1]], axis=1)

        else:
            x_f = T.matrix()
            x_b = T.matrix()

        if embedding == 'theano_word_embeddings':
            print('self.emb.shape:', self.emb.shape)

        #define symbolic variables
        lr = T.scalar('lr', dtype=theano.config.floatX)
        rho = T.scalar('rho', dtype=theano.config.floatX)
        t = T.iscalar()
        mom = T.scalar('mom', dtype=theano.config.floatX)

        print('W_xh_f.shape:', self.W_xh_f.shape)
        print('W_xh_b.shape:', self.W_xh_b.shape)
        print('W_hh_bi.shape:', self.W_hh_bi.shape)
        print('W_hy_f.shape:',self.W_hy_f.shape)
        print('b_hh_f.shape:', self.b_hh_f.shape)
        print('b_hy_f.shape:', self.b_hy_f.shape)
        print('W_hy_b.shape:',self.W_hy_b.shape)
        print('b_hh_b.shape:', self.b_hh_b.shape)
        print('b_hy_b.shape:', self.b_hy_b.shape)
        print('W_hy_bi.shape:', self.W_hy_bi.shape)
        print('b_hy_bi.shape:', self.b_hy_bi.shape)
        print('W_hy_bi.shape:', self.W_hy_bi.shape)

        if w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            print('W_hh.shape:',self.W_hh.shape)
            self.W_hh = theano.shared(self.W_hh, 'W_hh')
        else:
            print('W_hh_f.shape:',self.W_hh_f.shape)
            print('W_hh_b.shape:',self.W_hh_b.shape)
            self.W_hh_f = theano.shared(self.W_hh_f, 'W_hh_f')
            self.W_hh_b = theano.shared(self.W_hh_b, 'W_hh_b')

        self.W_xh_f = theano.shared(self.W_xh_f, 'W_xh_f')
        self.W_xh_b = theano.shared(self.W_xh_b, 'W_xh_b')
        #forward propagation parameters

        #self.W_hy_f = theano.shared(self.W_hy_f, 'W_hy_f')
        self.b_hh_f = theano.shared(self.b_hh_f, 'b_hh_f')
        #self.b_hy_f = theano.shared(self.b_hy_f, 'b_hy_f')
        #backward propagation parameters
        #self.W_hy_b = theano.shared(self.W_hy_b, 'W_hy_b')
        self.b_hh_b = theano.shared(self.b_hh_b, 'b_hh_b')
        #self.b_hy_b = theano.shared(self.b_hy_b, 'b_hy_b')
        #bidirection parameters
        self.W_hy_bi = theano.shared(self.W_hy_bi, 'W_hy_bi')
        self.b_hy_bi = theano.shared(self.b_hy_bi, 'b_hy_bi')
        #initital hidden state
        self.h0_tm1_f = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'h0_tm1_f')
        self.h0_tm1_b = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'h0_tm1_b')
        self.h0_bi_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'h0_bi_tm1')

        if w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            self.params = [ self.W_xh_f, self.W_xh_b, self.W_hh, self.W_hy_bi, self.b_hh_f, self.b_hy_bi, self.h0_tm1_f,
                            self.h0_tm1_b, self.b_hh_b, self.h0_bi_tm1 ]
            self.names  = ['W_xh_f', 'W_xh_b','W_hh', 'W_hy_bi', 'b_hh_f', 'b_hy_bi', 'h0_tm1_f',
                           'h0_tm1_b', 'b_hh_b', 'h0_bi_tm1' ]
        else:
            self.params = [ self.W_xh_f, self.W_xh_b, self.W_hh_f, self.W_hy_bi, self.b_hh_f, self.b_hy_bi, self.h0_tm1_f, self.W_hh_b,
                            self.h0_tm1_b, self.b_hh_b, self.h0_bi_tm1]
            self.names  = ['W_xh_f', 'W_xh_b', 'W_hh_f', 'W_hy_bi', 'b_hh_f', 'b_hy_bi', 'h0_tm1_f',
                           'W_hh_b', 'h0_tm1_b', 'b_hh_b', 'h0_bi_tm1' ]

        if embedding == 'theano_word_embeddings' or embedding == 'word2vec_update':
            self.params.append(self.emb)
            self.names.append('embeddings')

            if self.pos_feat_embedding == True:
                if self.pos_emb_type == 'COUPLED':
                    self.params.append(self.pos_emb_e1_e2)
                    self.names.append('pos_emb_e1_e2')
                elif self.pos_emb_type == 'DECOUPLED':
                    self.params.append(self.pos_emb_e1)
                    self.names.append('pos_emb_e1')
                    self.params.append(self.pos_emb_e2)
                    self.names.append('pos_emb_e2')

            elif self.ent_pres_feat_embedding == True:
                if self.ent_pres_emb_type == 'COUPLED':
                    self.params.append(self.ent_pres_emb_e1_e2)
                    self.names.append('ent_pres_emb_e1_e2')
                elif self.ent_pres_emb_type == 'DECOUPLED':
                    self.params.append(self.ent_pres_emb_e1)
                    self.names.append('ent_pres_emb_e1')
                    self.params.append(self.ent_pres_emb_e2)
                    self.names.append('ent_pres_emb_e2')

            if self.postag == True and self.update_pos_emb == True:
                self.params.append(self.postag_emb)
                self.names.append('postag_emb')

            if self.entity_class == True and self.update_ner_emb == True:
                self.params.append(self.entity_class_emb)
                self.names.append('entity_class_emb')

            if self.add_subtree_emb:
                # < to do >
                # add h0 of rnn and sub tree embedding to params
                if treernn_weights == 'independent':
                    self.W_hx = np.asarray(
                    rng.normal(size=(self.hidden_dim, self.emb_dim), scale =.01, loc=0.0), dtype = theano.config.floatX)
                    self.W_hh = np.asarray(
                                              rng.normal(size=(self.hidden_dim, self.hidden_dim), scale =.01, loc=0.0), dtype = theano.config.floatX)
                    self.b_h = np.zeros(self.hidden_dim, dtype=theano.config.floatX)
                    print('W_hx.shape:', self.W_hx.shape)
                    print('W_hh.shape:', self.W_hh.shape)
                    print('b_h.shape:', self.b_h.shape)
                    self.W_hx = theano.shared(self.W_hx, 'W_hx')
                    self.W_hh = theano.shared(self.W_hh, 'W_hh')
                    self.b_h = theano.shared(self.b_h, 'b_h')
                    self.params.extend([self.W_hx, self.W_hh, self.b_h]) # independent weights
                    self.names.extend(['W_hx', 'W_hh', 'b_h'])
                sdp_sent_aug_info_leaf_internal_x = T.ivector(name='sdp_sent_aug_info_leaf_internal_x')
                sdp_sent_aug_info_leaf_internal_x_cwords = T.imatrix(name='sdp_sent_aug_info_leaf_internal_x_cwords')  # word indices
                sdp_sent_aug_info_computation_tree_matrix = T.imatrix(name='sdp_sent_aug_info_computation_tree_matrix')  # shape [None, self.degree]
                sdp_sent_aug_info_output_tree_state_idx = T.ivector(name='sdp_sent_aug_info_output_tree_state_idx')
                self.num_words = sdp_sent_aug_info_leaf_internal_x.shape[0]  # total number of nodes (leaves + internal) in tree
                emb_x = self.emb[sdp_sent_aug_info_leaf_internal_x_cwords].reshape((
                    sdp_sent_aug_info_leaf_internal_x.shape[0], self.dim_emb*self.context_win_size))
                # emb_x = emb_x * T.neq(sdp_sent_aug_info_leaf_internal_x, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings
                emb_x = emb_x * T.neq(sdp_sent_aug_info_leaf_internal_x, -1).dimshuffle(0, 'x')
                if self.postag == True:
                    emb_x = theano.tensor.concatenate([emb_x, T.zeros((
                        self.num_words, self.dim_postag_emb),
                        dtype=theano.config.floatX)], axis=1)
                if self.entity_class == True:
                    emb_x = theano.tensor.concatenate([emb_x, T.zeros((
                        self.num_words, self.dim_entity_class_emb),
                        dtype=theano.config.floatX)], axis=1)
                self.tree_states = self.compute_tree(emb_x, sdp_sent_aug_info_computation_tree_matrix)
                self.final_state = self.tree_states[-1]

                # extract the values from output tree states
                aug_tree_emb_val = self.tree_states[sdp_sent_aug_info_output_tree_state_idx]
                x_f = theano.tensor.concatenate([x_f, aug_tree_emb_val], axis=1)
                x_b = theano.tensor.concatenate([x_b, aug_tree_emb_val[::-1]], axis=1)

        # network dynamics
        # use the scan operation, which allows to define loops
        # unchanging variables are passed into 'non-sequences', initialization occurs in 'outputs_info'
        # if we set outputs_info to None, this indicates to scan that it doesnt need to pass the prior result
        # to recurrent_fn.
        # The general order of function parameters to recurrent_fn is:
        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        # http://deeplearning.net/software/theano/library/scan.html
        if reload_model == True:
            self.load(reload_path)

        if w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            h_f, _ = theano.scan(self.recurrent_fn, sequences = x_f, outputs_info = [self.h0_tm1_f],
                                 non_sequences = [self.W_hh, self.W_xh_f, self.b_hh_f])

            # to investigate if h0_tm1_b should be assigned to h_f
            if self.w_hh_type == 'transpose':
                h_b, _ = theano.scan(self.recurrent_fn, sequences = x_b, outputs_info = [self.h0_tm1_b],
                                     non_sequences = [theano.tensor.transpose(self.W_hh), self.W_xh_b, self.b_hh_b])
            else:
                h_b, _ = theano.scan(self.recurrent_fn, sequences = x_b, outputs_info = [self.h0_tm1_b],
                                     non_sequences = [self.W_hh, self.W_xh_b, self.b_hh_b])

            def concat(h_f, h_b, h0_bi_tm1, W_hh_bi):
                h_t_bi = self.activ(h_f + h_b + T.dot(h0_bi_tm1, W_hh_bi))
                return h_t_bi

            h_bi, _ = theano.scan(fn=concat, sequences=[h_f,h_b], outputs_info=[self.h0_bi_tm1],
                                  non_sequences = [self.W_hh_bi])
        else:
            h_f, _ = theano.scan(self.recurrent_fn, sequences = x_f, outputs_info = [self.h0_tm1_f],
                                 non_sequences = [self.W_hh_f, self.W_xh_f, self.b_hh_f])

            h_b, _ = theano.scan(self.recurrent_fn, sequences = x_b, outputs_info = [self.h0_tm1_b],
                                 non_sequences = [self.W_hh_b, self.W_xh_b, self.b_hh_b])

            def concat(h_f, h_b, h0_bi_tm1, W_hh_bi):
                h_t_bi = self.activ(h_f + h_b + T.dot(h0_bi_tm1, W_hh_bi))
                return h_t_bi

            h_bi, _ = theano.scan(fn=concat, sequences=[h_f,h_b], outputs_info=[self.h0_bi_tm1],
                                  non_sequences = [self.W_hh_bi])

        # TO DO: to investigate use_dropout inside if statement after h_f and h_b
        if self.use_dropout == True:
            h_bi = self.dropout_layer(h_bi, use_noise, trng)

        #netwrok output
        self.y = T.dot(h_bi[-1], self.W_hy_bi) + self.b_hy_bi
        self.p_y_given_x = T.nnet.softmax(self.y)
        # compute prediction as class whose probability is maximal
        y_pred = T.argmax(self.p_y_given_x, axis=-1)
        y_pred_prob = self.p_y_given_x
        #computinhg cost
        #cost = -T.mean(T.log(self.p_y_given_x)[T.arange(t.shape[0]), t ])
        cost = -T.mean(T.log(self.p_y_given_x)[:, t ])

        if w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            cost +=  self.L2_reg * (T.sum(self.W_xh_f ** 2)
                                    + T.sum(self.W_xh_b ** 2)
                                    + T.sum(self.W_hh ** 2)
                                    + T.sum(self.W_hy_bi ** 2)
                                    + T.sum(self.h0_tm1_b ** 2)
                                    + T.sum(self.h0_tm1_f ** 2)
                                    + T.sum(self.h0_bi_tm1 ** 2))
        else:
            cost += self.L2_reg * (T.sum(self.W_xh_f ** 2)
                                   + T.sum(self.W_xh_b ** 2)
                                   + T.sum(self.W_hh_f ** 2)
                                   + T.sum(self.W_hy_bi ** 2)
                                   + T.sum(self.W_hh_b ** 2)
                                   + T.sum(self.h0_tm1_f ** 2)
                                   + T.sum(self.h0_tm1_b ** 2)
                                   + T.sum(self.h0_bi_tm1 ** 2))

        # the actual gradient descent, we need to evaluate the derivative of the cost function w.r.t. the parameters
        # and update their values. In the case of a recurrent network, this procedure is known as backpropagation
        # through time. Luckily with theano we dont really need to worry about this. Because all the code has been
        # defined as symbolic operations we can just ask for the derivatives of the parameters and it will propagate
        # them through the scan operation automatically

        updates, _, _, _, _ = create_optimization_updates(cost=cost, params=self.params, names=self.names,
                                                          method=self.optimiser, gradients=None,
                                                          lr=lr, rho=rho, max_norm=max_norm,
                                                          mom=mom)

        if embedding == 'theano_word_embeddings' or embedding == 'word2vec_update':
            if position_feat == True and entity_presence_feat == True:
                if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat, pos_feat_idx,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat, pos_feat_idx,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat, ent_pres_feat,
                                                       pos_feat_idx, ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.pos_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat, pos_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat, pos_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat, ent_pres_feat,
                                                       pos_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                elif self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat, ent_pres_feat,
                                                       ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                else:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat], y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat, ent_pres_feat], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

            elif entity_presence_feat == True:
                if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, ent_pres_feat, pos_feat_idx,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, ent_pres_feat, pos_feat_idx,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, ent_pres_feat,
                                                       pos_feat_idx, ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.pos_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, ent_pres_feat, pos_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, ent_pres_feat, pos_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, ent_pres_feat,
                                                       pos_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                elif self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, ent_pres_feat,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, ent_pres_feat,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, ent_pres_feat,
                                                       ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                else:
                    self.predict_prob = theano.function([idx_f, idx_b,ent_pres_feat], y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, ent_pres_feat], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, ent_pres_feat], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

            elif position_feat == True:
                if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, pos_feat_idx,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, pos_feat_idx,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat,
                                                       pos_feat_idx, ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.pos_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, pos_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, pos_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat,
                                                       pos_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                elif self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat,
                                                       ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                else:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat], y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

            else:
                if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat_idx,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat_idx,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom,
                                                       pos_feat_idx, ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.pos_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom,
                                                       pos_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                elif self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom,
                                                       ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.add_subtree_emb == True:
                    if self.postag == True and self.entity_class == True:
                        self.predict_prob = theano.function([idx_f, idx_b,
                                                             sdp_sent_aug_info_leaf_internal_x,
                                                             sdp_sent_aug_info_leaf_internal_x_cwords,
                                                             sdp_sent_aug_info_computation_tree_matrix,
                                                             sdp_sent_aug_info_output_tree_state_idx,
                                                             postag_idx, entity_class_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b,
                                                         sdp_sent_aug_info_leaf_internal_x,
                                                         sdp_sent_aug_info_leaf_internal_x_cwords,
                                                         sdp_sent_aug_info_computation_tree_matrix,
                                                         sdp_sent_aug_info_output_tree_state_idx,
                                                         postag_idx, entity_class_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b,
                                                           sdp_sent_aug_info_leaf_internal_x,
                                                           sdp_sent_aug_info_leaf_internal_x_cwords,
                                                           sdp_sent_aug_info_computation_tree_matrix,
                                                           sdp_sent_aug_info_output_tree_state_idx,
                                                           postag_idx, entity_class_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)

                    elif self.postag == True:
                        self.predict_prob = theano.function([idx_f, idx_b,
                                                             sdp_sent_aug_info_leaf_internal_x,
                                                             sdp_sent_aug_info_leaf_internal_x_cwords,
                                                             sdp_sent_aug_info_computation_tree_matrix,
                                                             sdp_sent_aug_info_output_tree_state_idx,
                                                             postag_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b,
                                                         sdp_sent_aug_info_leaf_internal_x,
                                                         sdp_sent_aug_info_leaf_internal_x_cwords,
                                                         sdp_sent_aug_info_computation_tree_matrix,
                                                         sdp_sent_aug_info_output_tree_state_idx,
                                                         postag_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b,
                                                           sdp_sent_aug_info_leaf_internal_x,
                                                           sdp_sent_aug_info_leaf_internal_x_cwords,
                                                           sdp_sent_aug_info_computation_tree_matrix,
                                                           sdp_sent_aug_info_output_tree_state_idx,
                                                           postag_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)

                    elif self.entity_class == True:
                        self.predict_prob = theano.function([idx_f, idx_b,
                                                             sdp_sent_aug_info_leaf_internal_x,
                                                             sdp_sent_aug_info_leaf_internal_x_cwords,
                                                             sdp_sent_aug_info_computation_tree_matrix,
                                                             sdp_sent_aug_info_output_tree_state_idx,
                                                             entity_class_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b,
                                                         sdp_sent_aug_info_leaf_internal_x,
                                                         sdp_sent_aug_info_leaf_internal_x_cwords,
                                                         sdp_sent_aug_info_computation_tree_matrix,
                                                         sdp_sent_aug_info_output_tree_state_idx,
                                                         entity_class_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b,
                                                           sdp_sent_aug_info_leaf_internal_x,
                                                           sdp_sent_aug_info_leaf_internal_x_cwords,
                                                           sdp_sent_aug_info_computation_tree_matrix,
                                                           sdp_sent_aug_info_output_tree_state_idx,
                                                           entity_class_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)
                    else:
                        self.predict_prob = theano.function([idx_f, idx_b,
                                                             sdp_sent_aug_info_leaf_internal_x,
                                                             sdp_sent_aug_info_leaf_internal_x_cwords,
                                                             sdp_sent_aug_info_computation_tree_matrix,
                                                             sdp_sent_aug_info_output_tree_state_idx],
                                                            y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b,
                                                         sdp_sent_aug_info_leaf_internal_x,
                                                         sdp_sent_aug_info_leaf_internal_x_cwords,
                                                         sdp_sent_aug_info_computation_tree_matrix,
                                                         sdp_sent_aug_info_output_tree_state_idx],
                                                        y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b,
                                                           sdp_sent_aug_info_leaf_internal_x,
                                                           sdp_sent_aug_info_leaf_internal_x_cwords,
                                                           sdp_sent_aug_info_computation_tree_matrix,
                                                           sdp_sent_aug_info_output_tree_state_idx,
                                                           t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)
                else:
                    if self.postag == True and self.entity_class == True:
                        self.predict_prob = theano.function([idx_f, idx_b, postag_idx, entity_class_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b, postag_idx, entity_class_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b, postag_idx, entity_class_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)

                    elif self.postag == True:
                        self.predict_prob = theano.function([idx_f, idx_b, postag_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b, postag_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b, postag_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)

                    elif self.entity_class == True:
                        self.predict_prob = theano.function([idx_f, idx_b, entity_class_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b, entity_class_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b, entity_class_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)
                    else:
                        self.predict_prob = theano.function([idx_f, idx_b], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)

            self.normalize = theano.function( inputs = [],
                                              updates = {self.emb: \
                                                             self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

        else:
            self.predict_prob = theano.function([x_f, x_b], y_pred_prob,
                                on_unused_input='ignore',
                                allow_input_downcast=True)
            self.classify = theano.function([x_f, x_b], y_pred,
                                on_unused_input='ignore',
                                allow_input_downcast=True)
            #the update itself happens directly on the parameter variables as part of theano update mechanis
            self.train_step = theano.function([x_f, x_b, t, lr, rho, mom], cost,
                                on_unused_input='ignore',
                                updates=updates,
                                allow_input_downcast=True)
    # recurrent step
    # The general order of function parameters to recurrent_fn is:
    # sequences (if any), prior result(s) (if needed), non-sequences (if any)
    # 'h_tm1': is the prior and would contain the output also for the intermediate computation which will be used in
    # the following computation as prior.

    def recurrent_fn(self, u_t, h_tm1, W_hh, W_xh, b_hh):
        h_t = self.activ(T.dot(h_tm1, W_hh) + T.dot(u_t, W_xh) + b_hh)
        return h_t

    def params(self):
        return self.params

    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of time steps (call it T) in the sequence
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y ])

    def save(self, folder):
        for param in self.params:
            np.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(np.load(os.path.join(folder,
                            param.name + '.npy')))

    '''
    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()
    '''

    # to sclae down the gradient
    # to deal with 'exploding gradient' probelm in RNN optimization
    def clip_norm(self, grad, norm):
        grad = T.switch(T.ge(norm, self.clip_norm_cutoff), grad*self.clip_norm_cutoff/norm, grad)
        return grad

    def get_gredient_clip(self, grads):
        # get the norm
        if self.clip_norm_cutoff > 0:
            norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
            grads = [self.clip_norm(g, norm) for g in grads]
        return grads

    def dropout_layer(self, state_before, use_noise, trng):
        proj = T.switch(use_noise,
                        (state_before*trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)),
                             state_before * 0.5)
        return proj

    def ortho_weight(self, ndim):
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype(config.floatX)

    def relu(x):
        return x * (x > 1e-6)


    def clip_relu(x, clip_lim=20):
        return x * (T.lt(x, 1e-6) and T.gt(x, clip_lim))


    def dropout(random_state, X, keep_prob=0.5):
        if keep_prob > 0. and keep_prob < 1.:
            seed = random_state.randint(2 ** 30)
            srng = RandomStreams(seed)
            mask = srng.binomial(n=1, p=keep_prob, size=X.shape,
                                dtype=theano.config.floatX)
            return X * mask
        return X


    def fast_dropout(random_state, X):
        seed = random_state.randint(2 ** 30)
        srng = RandomStreams(seed)
        mask = srng.normal(size=X.shape, avg=1., dtype=theano.config.floatX)
        return X * mask

    def _norm_constraint(self, param, update_step, max_col_norm):
        stepped_param = param + update_step
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, max_col_norm)
            scale = desired_norms / (1e-7 + col_norms)
            new_param = param * scale
            new_update_step = update_step * scale
        else:
            new_param = param
            new_update_step = update_step
        return new_param, new_update_step

    # def init_matrix(self, shape):
    #     return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)
    #
    # def init_vector(self, shape):
    #     return np.zeros(shape, dtype=theano.config.floatX)

    def create_recursive_unit(self):
        def unit(parent_x, child_h, child_exists):  # very simple
            h_tilde = T.sum(child_h, axis=0)
            if self.treernn_weights == 'independent':
                h = T.tanh(self.b_h + T.dot(self.W_hx, parent_x) + T.dot(self.W_hh, h_tilde))
            elif self.treernn_weights == 'shared':
                h = T.tanh(self.b_hh_f + T.dot(self.W_xh_f.T, T.concatenate([parent_x, h_tilde])) + T.dot(self.W_hh_bi, h_tilde))
            return h
        return unit

    def create_leaf_unit(self):
        rng = np.random.RandomState(1234)
        dummy = 0 * theano.shared(value=np.asarray(rng.normal(size=(self.max_degree, self.hidden_dim), scale =.01, loc=0.0) , dtype=theano.config.floatX))
        def unit(leaf_x):
            return self.recursive_unit(leaf_x, dummy, dummy.sum(axis=1))
        return unit

    def compute_tree(self, emb_x, tree):
        self.irregular_tree = False
        self.recursive_unit = self.create_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # compute leaf hidden states
        leaf_h, _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        init_node_h = leaf_h

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, node_h, last_h):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h = self.recursive_unit(cur_emb, child_h, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            return node_h[1:], parent_h

        dummy = theano.shared(np.zeros(self.hidden_dim, dtype=theano.config.floatX))
        (_, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, dummy],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)