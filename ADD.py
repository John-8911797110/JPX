import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.autograd import Function
import torch.nn.functional as F





class ADD:
    
    def __int__(self):
        pass

    def gen_market_label(self, df, raw_label):
        market_label = raw_label.groupby("datetime").mean().squeeze()
        bins = [-np.inf, self.lo, self.hi, np.inf]
        market_label = pd.cut(market_label, bins, labels=False)
        market_label.name = ("market_return", "market_return")
        df = df.join(market_label)
        return df

    @staticmethod
    def cal_ic_metrics(pred, label):
        metrics = {}
        metrics["mse"] = -F.mse_loss(pred, label).item()
        metrics["loss"] = metrics["mse"]
        pred = pd.Series(pred.cpu().detach().numpy())
        label = pd.Series(label.cpu().detach().numpy())
        metrics["ic"] = pred.corr(label)
        metrics["ric"] = pred.corr(label, method="spearman")
        return metrics

    def fit_thresh(self, train_label):
        market_label = train_label.groupby("datetime").mean().squeeze()
        self.lo, self.hi = market_label.quantile([1 / 3, 2 / 3])


class ADDModel(nn.Module):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        dec_dropout=0.5,
        base_model="GRU",
        gamma=0.1,
        gamma_clip=0.4,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.base_model = base_model
        if base_model == "GRU":
            self.enc_excess, self.enc_market = [
                nn.GRU(
                    input_size=d_feat,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        elif base_model == "LSTM":
            self.enc_excess, self.enc_market = [
                nn.LSTM(
                    input_size=d_feat,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        self.dec = Decoder(d_feat, 2 * hidden_size, num_layers, dec_dropout, base_model)

        ctx_size = hidden_size * num_layers
        self.pred_excess, self.adv_excess = [
            nn.Sequential(nn.Linear(ctx_size, ctx_size), nn.BatchNorm1d(ctx_size), nn.Tanh(), nn.Linear(ctx_size, 1))
            for _ in range(2)
        ]
        self.adv_market, self.pred_market = [
            nn.Sequential(nn.Linear(ctx_size, ctx_size), nn.BatchNorm1d(ctx_size), nn.Tanh(), nn.Linear(ctx_size, 3))
            for _ in range(2)
        ]
        self.before_adv_market, self.before_adv_excess = [RevGrad(gamma, gamma_clip) for _ in range(2)]

    def forward(self, x):
        x = x.reshape(len(x), self.d_feat, -1)
        N = x.shape[0]
        T = x.shape[-1]
        x = x.permute(0, 2, 1)

        out, hidden_excess = self.enc_excess(x)
        out, hidden_market = self.enc_market(x)
        if self.base_model == "LSTM":
            feature_excess = hidden_excess[0].permute(1, 0, 2).reshape(N, -1)
            feature_market = hidden_market[0].permute(1, 0, 2).reshape(N, -1)
        else:
            feature_excess = hidden_excess.permute(1, 0, 2).reshape(N, -1)
            feature_market = hidden_market.permute(1, 0, 2).reshape(N, -1)
        predicts = {}
        predicts["excess"] = self.pred_excess(feature_excess).squeeze(1)
        predicts["market"] = self.pred_market(feature_market)
        predicts["adv_market"] = self.adv_market(self.before_adv_market(feature_excess))
        predicts["adv_excess"] = self.adv_excess(self.before_adv_excess(feature_market).squeeze(1))
        if self.base_model == "LSTM":
            hidden = [torch.cat([hidden_excess[i], hidden_market[i]], -1) for i in range(2)]
        else:
            hidden = torch.cat([hidden_excess, hidden_market], -1)
        x = torch.zeros_like(x[:, 0, :])
        reconstructed_feature = []
        for i in range(T):
            x, hidden = self.dec(x, hidden)
            reconstructed_feature.append(x)
        reconstructed_feature = torch.stack(reconstructed_feature, 1)
        predicts["reconstructed_feature"] = reconstructed_feature
        return predicts


class Decoder(nn.Module):
    def __init__(self, d_feat=6, hidden_size=128, num_layers=1, dropout=0.5, base_model="GRU"):
        super().__init__()
        self.base_model = base_model
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.fc = nn.Linear(hidden_size, d_feat)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        output, hidden = self.rnn(x, hidden)
        output = output.squeeze(1)
        pred = self.fc(output)
        return pred, hidden


class RevGradFunc(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class RevGrad(nn.Module):
    def __init__(self, gamma=0.1, gamma_clip=0.4, *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.gamma_clip = torch.tensor(float(gamma_clip), requires_grad=False)
        self._alpha = torch.tensor(0, requires_grad=False)
        self._p = 0

    def step_alpha(self):
        self._p += 1
        self._alpha = min(
            self.gamma_clip, torch.tensor(2 / (1 + math.exp(-self.gamma * self._p)) - 1, requires_grad=False)
        )

    def forward(self, input_):
        return RevGradFunc.apply(input_, self._alpha)