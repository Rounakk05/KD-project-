import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

@torch.no_grad()
def feat_dim(backbone, size):
    was_training = backbone.training
    backbone.eval()
    d = backbone(torch.zeros(1, 3, size, size)).shape[1]
    backbone.train(was_training)
    return d

class Net(nn.Module):
    """timm backbone (pooled features) + linear classifier. forward -> (features, logits)."""

    def __init__(self, arch, num_classes, pretrained, img_size):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=0)
        if "swin" in arch or "vit" in arch:
            kw["img_size"] = img_size
        self.backbone = timm.create_model(arch, **kw)
        self.dim = feat_dim(self.backbone, img_size)
        self.fc = nn.Linear(self.dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return f, self.fc(f)

class Projector(nn.Module):
    """Maps a modality-specific feature into the shared semantic space."""

    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2), 
            nn.GELU(), 
            nn.Linear(out_dim * 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class ConfidenceGate(nn.Module):
    """Predicts a reliability score in (0,1) for EACH dimension of the teacher's
    aligned feature. Low score = feature is suppressed in the transfer loss."""

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, dim)
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z))

class KDHeads(nn.Module):
    """Everything that exists only during training (thrown away at inference)."""

    def __init__(self, method, ds, dt, num_classes, dim=256):
        super().__init__()
        self.method = method
        self.use_align = method in ("align", "align_conf", "full")
        self.use_sample_conf = method in ("align_conf", "full")
        self.use_gate = method == "full"
        if method == "feat_kd":
            self.reg = nn.Linear(ds, dt)
        if self.use_align:
            self.ps, self.pt = Projector(ds, dim), Projector(dt, dim)
            self.head = nn.Linear(dim, num_classes)  # ONE classifier shared by both modalities
        if self.use_gate:
            self.gate = ConfidenceGate(dim)

    def forward(self, fs, ft, ls, lt, y, args):
        fs, ft, ls, lt = fs.float(), ft.float(), ls.float(), lt.float()
        logs = {}
        ce = F.cross_entropy(ls, y)

        # --- sample-level confidence: how much does the teacher believe the TRUE class?
        if self.use_sample_conf:
            w = F.softmax(lt, 1).gather(1, y[:, None]).squeeze(1).detach()
            if args.conf_thresh > 0:
                w = w * (w >= args.conf_thresh).float()
        else:
            w = torch.ones_like(y, dtype=torch.float)
        logs["w_mean"] = w.mean().item()

        # --- logit KD (Hinton): student mimics teacher's softened probabilities
        kl = F.kl_div(F.log_softmax(ls / args.T, 1), F.softmax(lt / args.T, 1), reduction="none").sum(1) * args.T ** 2
        total = ce + args.alpha * (w * kl).mean()
        logs["ce"], logs["kd"] = ce.item(), (w * kl).mean().item()

        # --- naive feature KD baseline: regress teacher feature directly (no alignment)
        if self.method == "feat_kd":
            feat_loss = F.mse_loss(self.reg(fs), ft)
            total = total + args.beta * feat_loss
            logs["feat"] = feat_loss.item()

        # --- semantic alignment (+ optional feature gate)
        if self.use_align:
            zs, zt = self.ps(fs), self.pt(ft)
            zs_n, zt_n = F.normalize(zs, dim=1), F.normalize(zt, dim=1)
            if self.use_gate:
                g = self.gate(zt_n)
                zt_sel = zt * g
                l_gate = (g.mean() - args.rho) ** 2  # keeps the gate from switching everything off
                logs["gate_mean"] = g.mean().item()
            else:
                g, zt_sel, l_gate = torch.ones_like(zt_n), zt, torch.zeros((), device=zt.device)
            l_align = (w * ((zs_n - zt_n) ** 2 * g).sum(1)).mean()
            # shared classifier forces the common space to be class-discriminative
            # (also prevents the trivial "everything maps to one point" collapse)
            l_sem = F.cross_entropy(self.head(zs), y) + F.cross_entropy(self.head(zt_sel), y)
            total = total + args.beta * l_align + args.gamma * l_sem + args.lam_gate * l_gate
            logs["align"], logs["sem"] = l_align.item(), l_sem.item()
        return total, logs
