import torch
import torch.nn.functional as F

class ADDLoss:

    def __int__(self, mu=0.05, d_feat=5):
        self.mu = mu
        self.d_feat = d_feat


    def loss_pre_excess(self, pred_excess, label_excess, record=None):
        mask = ~torch.isnan(label_excess)
        pre_excess_loss = F.mse_loss(pred_excess[mask], label_excess[mask])
        if record is not None:
            record["pre_excess_loss"] = pre_excess_loss.item()
        return pre_excess_loss

    def loss_pre_market(self, pred_market, label_market, record=None):
        pre_market_loss = F.cross_entropy(pred_market, label_market)
        if record is not None:
            record["pre_market_loss"] = pre_market_loss.item()
        return pre_market_loss

    def loss_pre(self, pred_excess, label_excess, pred_market, label_market, record=None):
        pre_loss = self.loss_pre_excess(pred_excess, label_excess, record) + self.loss_pre_market(
            pred_market, label_market, record
        )
        if record is not None:
            record["pre_loss"] = pre_loss.item()
        return pre_loss

    def loss_adv_excess(self, adv_excess, label_excess, record=None):
        mask = ~torch.isnan(label_excess)
        adv_excess_loss = F.mse_loss(adv_excess.squeeze()[mask], label_excess[mask])
        if record is not None:
            record["adv_excess_loss"] = adv_excess_loss.item()
        return adv_excess_loss

    def loss_adv_market(self, adv_market, label_market, record=None):
        adv_market_loss = F.cross_entropy(adv_market, label_market)
        if record is not None:
            record["adv_market_loss"] = adv_market_loss.item()
        return adv_market_loss

    def loss_adv(self, adv_excess, label_excess, adv_market, label_market, record=None):
        adv_loss = self.loss_adv_excess(adv_excess, label_excess, record) + self.loss_adv_market(
            adv_market, label_market, record
        )
        if record is not None:
            record["adv_loss"] = adv_loss.item()
        return adv_loss

    def __call__(self, x, preds, label_excess, label_market, record=None):
        loss = (
                self.loss_pre(preds["excess"], label_excess, preds["market"], label_market, record)
                + self.loss_adv(preds["adv_excess"], label_excess, preds["adv_market"], label_market, record)
                + self.mu * self.loss_rec(x, preds["reconstructed_feature"], record)
        )
        if record is not None:
            record["loss"] = loss.item()
        return loss

    def loss_rec(self, x, rec_x, record=None):
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        rec_loss = F.mse_loss(x, rec_x)
        if record is not None:
            record["rec_loss"] = rec_loss.item()
        return rec_loss