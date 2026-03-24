import torch
import torch.nn as nn
import torch.nn.functional as F


class UltraMinimalLossNoSpatial(nn.Module):
    def __init__(
        self,
        dist_weight=0.3,
        road_weight=2.0,
        target_spacing=10,
        use_adversarial=False,
        adv_log_var_clamp=1.5,
        empty_index=20,
        road_index=0,
    ):
        super().__init__()
        self.dist_w = float(dist_weight)
        self.road_w = float(road_weight)
        self.target_spacing = int(target_spacing)
        self.use_adv = bool(use_adversarial)
        self.adv_clamp = float(adv_log_var_clamp)
        self.empty_idx = int(empty_index)
        self.road_idx = int(road_index)

        if self.use_adv:
            self.log_var_adv = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_var_adv", torch.zeros(1))

    def mse_with_distribution_loss(self, pred, target):
        mse = F.mse_loss(pred, target)

        if pred.shape[1] > self.empty_idx:
            pred_poi = pred[:, :self.empty_idx, :, :]
            target_poi = target[:, :self.empty_idx, :, :]
        else:
            pred_poi = pred
            target_poi = target

        pred_dist = pred_poi.mean(dim=[0, 2, 3])
        target_dist = target_poi.mean(dim=[0, 2, 3])

        pred_dist = pred_dist / (pred_dist.sum() + 1e-8)
        target_dist = target_dist / (target_dist.sum() + 1e-8)

        kl = (target_dist * torch.log((target_dist + 1e-8) / (pred_dist + 1e-8))).sum()
        combined = mse + self.dist_w * kl
        return combined, mse, kl

    def unified_road_grid_loss(self, fine_pred, coarse_road, road_target):
        B, C, H, W = coarse_road.shape

        template = torch.zeros(B, H, W, device=coarse_road.device)
        for i in range(0, H, self.target_spacing):
            template[:, i, :] = 1.0
        for j in range(0, W, self.target_spacing):
            template[:, :, j] = 1.0

        road_2d = coarse_road.squeeze(1) if C == 1 else coarse_road[:, 0, :, :]

        if road_target.dim() == 4:
            road_target_2d = road_target.squeeze(1)
        else:
            road_target_2d = road_target

        weight = 1.0 + 2.0 * template
        bce = F.binary_cross_entropy(road_2d, road_target_2d.float(), weight=weight)

        poi_nonroad = fine_pred[:, 1:self.empty_idx, :, :].sum(dim=1) if fine_pred.shape[1] > self.empty_idx else fine_pred[:, 1:, :, :].sum(dim=1)
        road_mask = (road_2d > 0.3).float()
        exclusion = (poi_nonroad * road_mask).mean()

        combined = bce + 0.5 * exclusion
        return combined, bce, exclusion

    def forward_generator(self, pred, target, coarse_road, road_target, d_fake=None):
        losses = {}
        mse_dist, mse, kl = self.mse_with_distribution_loss(pred, target)
        losses["mse_dist"] = mse_dist
        losses["mse"] = mse
        losses["kl"] = kl

        road, road_bce, road_excl = self.unified_road_grid_loss(pred, coarse_road, road_target)
        losses["road"] = road
        losses["road_bce"] = road_bce
        losses["road_exclusion"] = road_excl

        total = mse_dist + self.road_w * road

        if self.use_adv and d_fake is not None:
            adv_loss = -d_fake.mean()
            losses["adv"] = adv_loss
            total += 0.5 * torch.exp(-self.log_var_adv) * adv_loss + 0.5 * self.log_var_adv

        losses["total"] = total
        return losses

    def forward_discriminator(self, d_real, d_fake):
        return d_fake.mean() - d_real.mean()

    def clamp_log_vars(self):
        if self.use_adv:
            with torch.no_grad():
                self.log_var_adv.clamp_(-self.adv_clamp, self.adv_clamp)
