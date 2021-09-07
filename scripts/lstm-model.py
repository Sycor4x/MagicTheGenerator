#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-07-19 (year-month-day)

"""
python scripts/lstm-model.py -i local_data/creatures-planeswalkers.json -b 128 -e 500 -r 1e-2
"""


import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.distributions.categorical import Categorical
from torch.linalg import vector_norm
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

card_data_fields = [
    "artist",
    "border",
    "cmc",
    "color_identity",
    "colors",
    "flavor",
    "foreign_names",
    "hand",
    "id",
    "image_url",
    "mana_cost",
    "multiverse_id",
    "name",
    "names",
    "number",
    "power",
    "rarity",
    "release_date",
    "set",
    "set_name",
    "source",
    "starter",
    "subtypes",
    "supertypes",
    "text",
    "timeshifted",
    "toughness",
    "type",
    "types",
    "variations",
    "watermark",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--infile", type=pathlib.Path, required=True, help="the file to read"
    )
    parser.add_argument("-r", "--lr", type=float, default=1e-3, help="NN learning rate")
    parser.add_argument("-e", "--n_epoch", type=int, default=1, help="NN epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-t", "--temperature", type=float, default=1., help="temperature for random sampling")
    parser.add_argument(
        "-T",
        "--test_run",
        action="store_true",
        help="set this flag to just use a tiny data set",
    )
    parser.add_argument("-d", "--decoder", choices=[ "viterbi", "random"], help="the algorithm to use for decoding the model")
    return parser.parse_args()


class Codec(object):
    def __init__(self, max_len=32):
        self.maxlen = max_len
        self.special = ["_PAD", "_GO", "_END", "_UNK"]
        self.str2num = {sg: ix for ix, sg in enumerate(self.special)}
        self.exclude = [self.str2num[c] for c in self.special]
        all_chars = """ "&'(),-./?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyzàáâíöúû"""
        for s in all_chars:
            self.str2num.update({s: len(self.str2num)})
        self.num2str = {ix: sg for sg, ix in self.str2num.items()}

    def __len__(self):
        return len(self.str2num)

    def encode_many(self, string_list):
        return [self.encoder(s) for s in string_list]

    def encoder(self, in_string, go_end=True):
        out = self.str2num["_PAD"] * np.ones(self.maxlen)
        if go_end:
            in_string = ["_GO", *in_string, "_END"]
        for i, c in enumerate(in_string):
            out[i] = self.str2num.get(c, self.str2num["_UNK"])
        return out

    def decode_many(self, arr_list):
        return [self.decoder(arr) for arr in arr_list]

    def decoder(self, arr):
        list_str = [self.num2str[ix] for ix in arr if ix not in self.exclude]
        return "".join(list_str)


def process_card_names(card_data, minlen=4):
    out = []
    for n in card_data:
        # skip double cards; long-term, work out how to incorporate them
        if "//" in n or len(n) < minlen:
            continue
        out.append(n)
    out = sorted(out, key=len, reverse=True)
    return out


class NameDataset(Dataset):
    def __init__(self, name_arr, feat_arr, weight):
        assert name_arr.shape[0] == feat_arr.shape[0]
        self.name_arr = name_arr
        self.feat_arr = feat_arr
        self.weight = weight

    def __getitem__(self, item):
        name = self.name_arr[item, :]
        x = name[:-1]
        y = name[1:]
        z = self.feat_arr[item, :]
        w = self.weight[item]
        return z, x, y, w

    def __len__(self):
        return self.name_arr.shape[0]


def partition_data(data, test=0.1, validate=0.1,stratify=None):
    assert test > 0.0
    assert validate > 0.0
    assert test + validate < 0.5
    (
        data__,
        data_test,
    ) = train_test_split(data, test_size=test, stratify=stratify)
    new_validate = validate / (1.0 - test)
    (
        data_train,
        data_valid,
    ) = train_test_split(data__, test_size=new_validate, stratify=stratify[data__])
    return data_train, data_valid, data_test


class UnitNorm(nn.Module):
    def __init__(self, ord=1, dim=None):
        super(UnitNorm, self).__init__()
        self.ord = ord
        self.dim = dim

    def forward(self, x):
        x = x + 1e-6
        norm = vector_norm(x, dim=self.dim, ord=self.ord, keepdim=True)
        return x / (1e-6 + norm)


class ResidualNet(nn.Module):
    def __init__(self, n_units=256, activation=nn.ELU()):
        super(ResidualNet, self).__init__()
        self.dense_net = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.BatchNorm1d(n_units),
            activation,
            nn.Linear(n_units, n_units),
            nn.BatchNorm1d(n_units),
            activation,
        )

    def forward(self, x):
        delta = self.dense_net(x)
        return delta + x


class InitializerNet(nn.Module):
    def __init__(self, n_features, n_units, lstm_layers=1, activation=nn.ELU()):
        super(InitializerNet, self).__init__()
        self.lstm_layers = lstm_layers
        self.n_units = n_units
        self.initial_net = [nn.Linear(n_features, n_units ) for _ in range( lstm_layers)]
        self.resid_net = [ResidualNet(n_units,activation=activation) for _ in range(lstm_layers)]
        self.norm = UnitNorm(dim=1)

    def forward(self, x):
        y = [self.initial_net[ix](x) for ix in range( self.lstm_layers)]
        z = [resid(inp) for resid, inp in zip(self.resid_net, y)]
        w = [self.norm(z_) for z_ in z]
        out = torch.stack(w, dim=0)
        return out


class NameNet(nn.Module):
    def __init__(self, vocab_size, n_features, dropout=0.0, n_units=256, lstm_layers=2):
        super(NameNet, self).__init__()
        self.char_emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=n_units,
            max_norm=1.0,
            scale_grad_by_freq=True,
            padding_idx=0,
        )
        self.hidden_state_net = InitializerNet(
            n_features=n_features, n_units=n_units, lstm_layers=lstm_layers
        )
        self.cell_state_net = InitializerNet(
            n_features=n_features, n_units=n_units, lstm_layers=lstm_layers
        )
        self.lstm_net = nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dense_net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(n_units, vocab_size, bias=True)
        )
        self.regression_net = nn.Sequential(
            nn.Dropout(dropout),
            ResidualNet(n_units=n_units),
            nn.BatchNorm1d(n_units),
            nn.Linear(n_units, n_features),
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, name, card_data=None, hc_state=None):
        assert hc_state is not None or card_data is not None
        if hc_state is None:
            h_0 = self.hidden_state_net(card_data)
            c_0 = self.cell_state_net(card_data)
        else:
            h_0, c_0 = hc_state
        name_emb = self.char_emb(name)
        lstm_out, hc_state = self.lstm_net(name_emb, (h_0, c_0))
        logits = self.dense_net(lstm_out)
        return logits.transpose(1, 2), hc_state

    def predict_proba(self, name, card_data=None, hc_state=None):
        logits, hc_state = self.forward(
            name=name, card_data=card_data, hc_state=hc_state
        )
        probs = self.softmax(logits)
        return probs, hc_state

    def predict_log_proba(self, name, card_data=None, hc_state=None):
        logits, hc_state = self.forward(
            name=name, card_data=card_data, hc_state=hc_state
        )
        log_probs = logits.exp() - logits.logsumexp(dim=2, keepdims=True)
        return log_probs, hc_state


class ColorEncoder(object):
    def __init__(self, card_info=None):
        self.color_dict = {c: i for i, c in enumerate("WUBRG")}
        self.color_dict[None] = len(self.color_dict)

    def encode(self, card_color_list):
        out = np.zeros(len(self.color_dict))
        if card_color_list is None:
            out[self.color_dict[None]] = 1.0
        else:
            out[[self.color_dict[c] for c in card_color_list]] = 1.0
        return out

    def encode_many(self, card_info):
        color_identity = dict()
        for name, card in card_info.items():
            col = card.get("color_identity", None)
            if col:
                col = "".join(sorted(col))
            else:
                col = None
            color_identity.update({name: self.encode(col)})
        return color_identity


class SubtypeEncoder(object):
    def __init__(self, card_info):
        subtype_set = set()
        for c in card_info.values():
            st_list = c.get("subtypes", [])
            if st_list:
                subtype_set |= set(st_list)
        self.subtypes_dict = dict()
        for st in sorted(subtype_set):
            if st not in self.subtypes_dict:
                self.subtypes_dict.update({st: len(self.subtypes_dict)})

    def encode(self, subtypes_list):
        out = np.zeros(len(self.subtypes_dict))
        if subtypes_list is None:
            return out
        out[[self.subtypes_dict[c] for c in subtypes_list]] = 1.0
        return out

    def __call__(self, card_info):
        out = dict()
        for name, card in card_info.items():
            subtypes = card.get("subtypes", [])
            if not subtypes:
                continue
            out.update({name: self.encode(subtypes)})
        return out


def get_weights(card_info):
    out = dict()
    for name, data in card_info.items():
        supertype = data.get("supertypes", [])
        card_type = data.get("types", [])
        supertype = [] if supertype is None else supertype
        card_type = [] if card_type is None else card_type
        wt = 1.0
        if "legendary" in supertype or "planeswalker" in card_type:
            wt = 10.0
        out.update({name: wt})
    return out


class Gatherer(object):
    def __init__(self, card_info):
        super(Gatherer, self).__init__()
        self.subtypes = SubtypeEncoder(card_info)
        self.colors = ColorEncoder(card_info)

    def __call__(self, card_info):
        out = dict()
        for name, info in card_info.items():
            color_id = self.colors.encode(info["color_identity"])
            subtypes = self.subtypes.encode(info["subtypes"])
            # cmc = np.array([info.get("cmc", 0)])
            arr = np.concatenate((color_id, subtypes))
            # arr = np.concatenate((cmc, arr))
            out.update({name: arr})
        return out


class RandomSampler(object):
    def __init__(self, codec, net, limit=35, temperature=1.0):
        assert 2 < limit < 128
        assert isinstance(limit, int)
        assert 0.0 < temperature
        self.codec = codec
        self.net = net
        self.limit = limit
        self.t = temperature

    @torch.no_grad()
    def __call__(self, card_info, argmax=False):
        if argmax:
            sampler = self.argmax_sampler
        else:
            sampler = self.random_sampler
        n = card_info.size(0)
        tokens = torch.LongTensor(np.zeros((n, self.limit + 1)))
        tokens[:, 0] = self.codec.str2num["_GO"]
        logits, hc_state = self.net(tokens[:, 0].view(-1, 1), card_data=card_info)
        # hc_state: (layers, observations, units)
        # TODO - we could add "early stopping" logic to terminate if all samples include
        #  the _END token
        #       but it's probably not worth the effort because sequences are short
        for i in range(0, self.limit):
            # while len(tokens) < self.limit and self.codec.decoder(new_token) != "_END":
            tokens[:, i + 1] = sampler(logits)
            logits, hc_state = self.net(tokens[:, i + 1].view(-1, 1), hc_state=hc_state)
        tokens = tokens.detach().numpy()
        extent = {k: self.limit for k in range(n)}
        for row, col in np.argwhere(tokens == self.codec.str2num["_END"]):
            char_idx = extent.get(row, self.limit)
            extent.update({row: min(col, char_idx)})
        out = self.codec.decode_many(tokens)
        out = [out[m][:p] for m, p in extent.items()]
        return out

    def random_sampler(self, logits):
        logits /= self.t
        new_tokens = Categorical(logits=logits.squeeze(2)).sample()
        return new_tokens

    def argmax_sampler(self, logits: torch.Tensor):
        return logits.argmax(1).view(-1)


class ViterbiSampler(object):
    def __init__(self, codec, net, limit=35):
        assert 2 < limit < 128
        assert isinstance(limit, int)
        self.codec = codec
        self.net = net
        self.limit = limit

    @staticmethod
    @torch.no_grad()
    def predict_viterbi_many(log_emiss, log_prior, log_trans):
        """
        Given the Viterbi algorithm and the estimated parameters of the model, compute the most
        likely sequence of hidden states states.
        This is designed to be a completely general case, allowing for time-varying
        transition matrices and per-observational-unit prior vectors.
        Alternatively, `predict_viterbi_single` is a Viterbi algorithm implementation that
        assumes the more typical HMM model where transitions and priors are fixed values.
        TODO - generalize this for the case of partially-observed latent states. It will just
               involve adding logic for a mask to be applied to the new alphas, as per the
               buam-welch modification.
        """
        # TODO - add coverage for partially-observed latent sequences.
        # TODO - complete this method
        # We work on the log-probability scale because all we care about is which state has
        # highest probability.
        N, T, K, _ = log_trans.shape
        assert (*log_emiss.shape,) == (N, T, K)
        assert (*log_prior.shape,) in ((N, K), (1, K), (K,))
        assert (*log_trans.shape,) == (N, T, K, K)

        log_alpha = torch.zeros((N, T, K))
        psi = torch.zeros_like(log_alpha, dtype=torch.long)
        # Compute the first alpha
        log_alpha0 = log_prior + log_emiss[:, 0, ...]
        log_alpha0 -= torch.logsumexp(log_alpha0, dim=1, keepdims=True)
        log_alpha[:, 0, :] = log_alpha0
        for t in range(1, T):
            prev_a = log_alpha[:, t - 1, :]  # (N, K)
            z = prev_a.unsqueeze(-1).repeat(1, 1, K)
            z += log_trans[:, t - 1, ...]  # (N, K, K)
            val, idx = z.max(1)
            psi[:, t, :] = idx
            alpha_t = val + log_emiss[:, t, :]
            alpha_t -= torch.logsumexp(alpha_t, dim=1, keepdims=True)
            log_alpha[:, t, :] = alpha_t

        _, idx = log_alpha[:, -1, :].max(1)
        path = torch.zeros((N, T), dtype=torch.long)
        path[:, -1] = idx
        buffer = torch.zeros(N, dtype=torch.long)
        for k in range(T - 2, -1, -1):
            child = path[:, k + 1]  # (N,)
            for i, j in enumerate(child):
                buffer[i] = psi[i, k + 1, j]
            path[:, k] = buffer
        return path

    @torch.no_grad()
    def __call__(self, card_info, argmax=False):
        N = card_info.size(0)
        T = self.limit + 1
        K = len(self.codec.str2num)

        log_prior = -100.0 * torch.ones((N, K))
        log_prior[:, self.codec.str2num["_GO"]] = 0.0

        log_trans = torch.zeros((N, T, K, K))
        log_trans[:, 0, :, :] = -100.0
        in_tokens = torch.LongTensor(self.codec.str2num["_GO"] * np.ones((N, 1)))
        log_prob, (h_state, c_state) = self.net.predict_log_proba(
            in_tokens, card_data=card_info
        )
        log_trans[:, 0, self.codec.str2num["_GO"], :] = log_prob.squeeze(2)

        # hc_state: (layers, observations, units)
        h_buff = h_state.repeat(K, 1, 1, 1)
        c_buff = c_state.repeat(K, 1, 1, 1)
        for i in range(1, T):
            # while len(tokens) < self.limit and self.codec.decoder(new_token) != "_END":
            for j, tok in enumerate(range(K)):
                in_tokens = torch.LongTensor(tok * np.ones((N, 1)))
                hc_state = (h_buff[j, ...], c_buff[j, ...])
                log_prob, (h_state, c_state) = self.net(in_tokens, hc_state=hc_state)
                log_trans[:, i, j, :] = log_prob.squeeze(2)
                h_buff[j, ...] = h_state
                c_buff[j, ...] = c_state

        log_emiss = torch.ones((N, T, K)) / K
        tokens = self.predict_viterbi_many(log_emiss, log_prior, log_trans )
        tokens = tokens.detach().numpy()
        out = self.codec.decode_many(tokens)
        return out


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    args = parse_args()
    with open(args.infile) as f:
        all_cards = json.load(f)
    cards = dict()
    for i, (k, v) in enumerate(all_cards.items()):
        leg = v.get("legalities", [])
        if leg is None:
            continue
        if any(
            [
                (y["format"] == "Vintage")
                and (y["legality"] in ["Legal", "Restricted"])
                for y in leg
            ]
        ):
            cards.update({k: v})
        if args.test_run and i > 100:
            break

    features = Gatherer(cards)(cards)

    names = process_card_names(cards)
    weights = np.vstack([get_weights(cards)[n] for n in names])
    features = np.vstack([features[n] for n in names])
    print(f"There are {len(names)} names.")
    maxlen = 2 + max(len(n) for n in names)
    print(f"The longest sequence is {maxlen} characters.")
    codec = Codec(max_len=maxlen)
    enc_names = np.vstack(codec.encode_many(names)).astype(np.int64)
    # TODO - add feature for legendary supertype
    train_idx, val_idx, test_idx = partition_data(
        range(len(names)), validate=0.15, test=0.15, stratify=[int(w > 5) for w in weights]
    )
    print(
        f"Train size: {len(train_idx)}; Valid. size: {len(val_idx)}; Test size: {len(test_idx)}."
    )

    zscorer = StandardScaler()
    train_feats = zscorer.fit_transform(features[train_idx, :])
    val_feats = zscorer.transform(features[val_idx, :])
    test_feats = zscorer.transform(features[test_idx, :])

    train_data = DataLoader(
        NameDataset(
            name_arr=enc_names[train_idx, :],
            feat_arr=train_feats,
            weight=weights[train_idx, :],
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True if not args.test_run else False,
    )
    validation_data = DataLoader(
        NameDataset(
            name_arr=enc_names[val_idx, :],
            feat_arr=val_feats,
            weight=weights[val_idx, :],
        ),
        batch_size=10000,
        shuffle=False,
        drop_last=False,
    )
    test_data = DataLoader(
        NameDataset(
            name_arr=enc_names[test_idx, :],
            feat_arr=test_feats,
            weight=weights[test_idx, :],
        ),
        batch_size=10000,
        shuffle=False,
        drop_last=False,
    )
    my_net = NameNet(
        vocab_size=len(codec),
        n_features=features.shape[1],
        n_units=64,
        lstm_layers=3,
        dropout=0.25,
    )
    if args.decoder == "viterbi":
        my_sampler = ViterbiSampler(codec=codec, net=my_net)
    elif args.decoder == "random":
        my_sampler = RandomSampler(codec=codec, net=my_net)
    else:
        raise ValueError(f"Value provided to argument `decoder` not recognized: {args.decoder}")
    param_count = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
    print(f"The model has {param_count} parameters.")
    my_optim = Adam(my_net.parameters(), lr=args.lr)
    my_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    wt_buf = np.zeros(len(train_data))
    buf = np.zeros(len(train_data))
    for i in range(args.n_epoch):
        my_net.train()
        for j, (feats, name_in, name_target, wt) in enumerate(train_data):
            my_optim.zero_grad()
            name_pred, *_ = my_net(name=name_in, card_data=feats)
            wt_loss = my_loss(name_pred, name_target) * wt
            wt_loss = wt_loss.sum() / wt.sum()
            wt_loss.backward()
            my_optim.step()
            unwt_loss = my_loss(name_pred, name_target).mean()
            wt_buf[j] = wt_loss.item()
            buf[j] = unwt_loss.item()
        my_net.eval()
        for val_feats, val_name, val_target, wt in validation_data:
            with torch.no_grad():
                val_pred, *_ = my_net(name=val_name, card_data=val_feats)
                val_loss = my_loss(val_pred, val_target).mean()
                generated_names = my_sampler(val_feats[:10])
        print(
            f"Epoch {i}\tTraining loss {buf.mean() :.6f} \u00b1 {1.96 * buf.std(ddof=1) / np.sqrt( buf.size):.6f}"
        )
        print(f"Epoch {i}\tValidation loss {val_loss:.6f}")
        print(generated_names)
