# PyTorch + Sinkhorn toy demo (GPU if available)
# This notebook-style script demonstrates:
# 1) Building a simple encoder that predicts N event "slots"
# 2) Computing a cost matrix between predicted slots and ground-truth events
# 3) Running a GPU-friendly Sinkhorn algorithm to obtain a soft assignment P (N x M)
# 4) Computing fieldwise losses using the soft assignment (present/type/regression)
#
# Notes / simplifications for clarity:
# - We use uniform marginals for Sinkhorn: row_sum = 1/N, col_sum = 1/M.
# - From the soft transport P, we build *soft targets* for present/type/time/mag.
# - This is a demonstration: in production you may want a dustbin/null column, adaptive marginals, or hard assignments.
#
# Run this cell to see losses and one training step on synthetic data.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random, os
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# Toy synthetic setup
# ---------------------------
torch.manual_seed(0)
random.seed(0)

B = 4             # batch size
T = 128           # length of time series (synthetic)
channels = 3
N_slots = 8       # number of predicted slots
M_max = 5         # maximum number of GT events per sample (variable <= M_max)

# ground-truth event types (for synthetic)
EVENT_TYPES = ["price_jump", "volume_spike", "trend"]
n_types = len(EVENT_TYPES)

# ---------------------------
# Simple encoder + heads
# ---------------------------
class SimpleEncoder(nn.Module):
    def __init__(self, channels, hidden=128, zdim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(hidden, zdim)
    def forward(self, x):
        # x: [B, channels, T]
        h = self.conv(x).squeeze(-1)   # [B, hidden]
        z = self.fc(h)                 # [B, zdim]
        return z

class SlotHeads(nn.Module):
    def __init__(self, zdim, N_slots, n_types):
        super().__init__()
        self.N = N_slots
        self.slot_mlp = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.ReLU(),
            nn.Linear(zdim, N_slots * 128),
        )
        # Per-slot decoders
        self.present_head = nn.Linear(128, 1)
        self.type_head = nn.Linear(128, n_types)
        self.time_head = nn.Linear(128, 1)
        self.mag_head = nn.Linear(128, 1)
        self.prio_head = nn.Linear(128, 1)
    def forward(self, z):
        # z: [B, zdim]
        slots = self.slot_mlp(z)                           # [B, N*slot_dim]
        slots = slots.view(z.size(0), self.N, 128)        # [B, N, 128]
        present_logit = self.present_head(slots).squeeze(-1)  # [B, N]
        type_logits = self.type_head(slots)               # [B, N, n_types]
        time = self.time_head(slots).squeeze(-1)          # [B, N]
        mag = self.mag_head(slots).squeeze(-1)            # [B, N]
        prio = torch.sigmoid(self.prio_head(slots).squeeze(-1)) # [B, N], in [0,1]
        return present_logit, type_logits, time, mag, prio

# Build model
encoder = SimpleEncoder(channels, hidden=128, zdim=256).to(device)
heads = SlotHeads(zdim=256, N_slots=N_slots, n_types=n_types).to(device)

# optimizer for demo
opt = torch.optim.AdamW(list(encoder.parameters()) + list(heads.parameters()), lr=1e-3)

# ---------------------------
# Sinkhorn implementation (entropy-regularized OT)
# ---------------------------
def sinkhorn_logspace(log_K: torch.Tensor, r: torch.Tensor, c: torch.Tensor, iters: int = 50) -> torch.Tensor:
    """
    log_K: [N, M] log of kernel K = exp(-C/eps) (we pass log to improve stability)
    r: [N] desired row sums (sum to 1)
    c: [M] desired col sums (sum to 1)
    returns P: [N, M] transport matrix whose row sums approx r and col sums approx c
    """
    # We'll work in log-domain to avoid underflow: use log u and log v updates
    N, M = log_K.shape
    log_r = torch.log(r + 1e-30).to(log_K.device)   # [N]
    log_c = torch.log(c + 1e-30).to(log_K.device)   # [M]
    # Initialize log_u = zeros(N), log_v = zeros(M)
    log_u = torch.zeros(N, device=log_K.device)
    log_v = torch.zeros(M, device=log_K.device)
    for _ in range(iters):
        # update log_u so that row sums match r: u = r / (K v)  -> log_u = log(r) - log(K @ v)
        # but log(K @ v) = logsumexp(log_K + log_v[None,:], dim=1)
        log_K_plus_v = log_K + log_v.unsqueeze(0)   # [N, M]
        log_Kv = torch.logsumexp(log_K_plus_v, dim=1)  # [N]
        log_u = log_r - log_Kv
        # update log_v so that col sums match c: v = c / (K^T u) -> log_v = log(c) - log(K^T u)
        log_K_plus_u = log_K + log_u.unsqueeze(1)   # [N, M]
        log_KTu = torch.logsumexp(log_K_plus_u, dim=0)  # [M]
        log_v = log_c - log_KTu
    # compute final P = diag(u) K diag(v) -> log_P = log_u[:,None] + log_K + log_v[None,:]
    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    P = torch.exp(log_P)
    return P

# ---------------------------
# Synthetic GT generator
# ---------------------------
def make_synthetic_batch(B, channels, T, M_max):
    # returns time series x [B,channels,T] and gt lists per sample (variable M)
    x = torch.randn(B, channels, T)
    gt_list = []
    for b in range(B):
        M = random.randint(1, M_max)
        gts = []
        for j in range(M):
            typ = random.randint(0, n_types - 1)
            t = random.uniform(0, T-1)         # time index (float)
            if typ == 0:   # price_jump
                mag = random.uniform(0.01, 0.05)
            elif typ == 1: # volume_spike
                mag = random.uniform(20000, 100000)
            else:
                mag = random.uniform(0.001, 0.02)
            priority = random.uniform(0.5, 1.0)
            gts.append({"type": typ, "time": t, "mag": mag, "prio": priority})
        gt_list.append(gts)
    return x, gt_list

# small helper to compute cost matrix between predicted slots and GTs (per sample)
def compute_cost_matrix(pred_time, pred_mag, pred_type_logits, pred_prio, gt):
    # pred_time, pred_mag: [N],
    # pred_type_logits: [N, n_types]
    # gt: list of dicts length M
    N = pred_time.shape[0]
    M = len(gt)
    # collect gt tensors
    gt_types = torch.tensor([g["type"] for g in gt], device=pred_time.device, dtype=torch.long)
    gt_times = torch.tensor([g["time"] for g in gt], device=pred_time.device, dtype=torch.float32)
    gt_mags = torch.tensor([g["mag"] for g in gt], device=pred_time.device, dtype=torch.float32)
    gt_prios = torch.tensor([g["prio"] for g in gt], device=pred_time.device, dtype=torch.float32)

    # time cost (L1) normalized by T
    time_cost = torch.abs(pred_time.unsqueeze(1) - gt_times.unsqueeze(0)) / T  # [N,M]
    # mag cost: normalize each gt by magnitude scale depending on type (simple heuristic)
    mag_scale = torch.where(gt_types == 1, 100000.0, 0.05)  # approximate scale per gt type (volume vs price)
    mag_cost = torch.abs(pred_mag.unsqueeze(1) - gt_mags.unsqueeze(0)) / (mag_scale.unsqueeze(0) + 1e-9)

    # type cost: use negative log prob of the gt type under pred softmax
    pred_type_logprob = F.log_softmax(pred_type_logits, dim=-1)  # [N, n_types]
    # expand to [N, M] gathering by gt type
    type_cost = -pred_type_logprob[:, gt_types]  # [N, M]

    # priority mismatch squared
    prio_cost = (pred_prio.unsqueeze(1) - gt_prios.unsqueeze(0)) ** 2  # [N,M]

    # present penalty: encourage matching to slots with high present probability
    # we won't include present here, we'll handle present with separate BCE using row sums later.

    # final weighted sum (weights are hyperparams)
    C = 1.0 * time_cost + 10.0 * mag_cost + 1.0 * type_cost + 1.0 * prio_cost
    return C, {"gt_types": gt_types, "gt_times": gt_times, "gt_mags": gt_mags, "gt_prios": gt_prios}

# ---------------------------
# Training step (single batch) demonstration
# ---------------------------
def training_step(batch_x, batch_gts):
    encoder.train(); heads.train()
    opt.zero_grad()
    x = batch_x.to(device)
    z = encoder(x)  # [B, zdim]
    present_logits, type_logits, time_pred, mag_pred, prio_pred = heads(z)
    # shapes: present_logits [B, N], type_logits [B, N, n_types], time_pred [B,N], mag_pred [B,N], prio_pred [B,N]

    total_loss = 0.0
    losses_details = {"bce_present":0.0, "type":0.0, "time":0.0, "mag":0.0, "prio":0.0}
    eps = 1e-8

    for b in range(B):
        # per-sample predictions
        p_logit = present_logits[b]           # [N]
        p_prob = torch.sigmoid(p_logit)       # [N]
        t_pred = time_pred[b]                 # [N]
        m_pred = mag_pred[b]                  # [N]
        ty_logits = type_logits[b]            # [N, n_types]
        prio = prio_pred[b]                   # [N]

        gt = batch_gts[b]
        M = len(gt)

        # compute cost matrix [N, M]
        C, gt_tensors = compute_cost_matrix(t_pred, m_pred, ty_logits, prio, gt)
        # convert cost to log_K = -C/eps for Sinkhorn
        eps_sink = 0.1
        log_K = (-C / eps_sink)
        # marginals r (rows) and c (cols) as uniform distributions
        r = torch.ones(N_slots, device=device) / float(N_slots)
        c = torch.ones(M, device=device) / float(M)
        P = sinkhorn_logspace(log_K, r, c, iters=60)   # P shape [N, M]
        # Now P is a soft transport matrix where rows approx sum to 1/N and cols to 1/M

        # derive soft targets and losses
        # present target: how much gt mass is assigned to this slot (scaled by M to represent count-like value)
        row_sum = P.sum(dim=1)                      # ~1/N each in uniform case, but scaled by M below
        present_target = row_sum * float(M)         # [N] soft count assigned to slot i
        # clamp to [0,1] for BCE target (since present is probability)
        present_target_clamped = torch.clamp(present_target, 0.0, 1.0)

        # BCE loss for present
        bce = F.binary_cross_entropy_with_logits(p_logit, present_target_clamped)
        losses_details["bce_present"] += bce.item()

        # type soft-target: for each slot, target distribution over types is weighted sum of GT one-hot types
        gt_type_onehot = F.one_hot(gt_tensors["gt_types"], num_classes=n_types).float()  # [M, n_types]
        # soft target per slot: sum_j P_ij * onehot_j  -> [N, n_types]
        type_target = (P @ gt_type_onehot) * float(M)   # scale back by M similar to present_target
        # Normalize so rows sum to <=1
        type_target = type_target / (type_target.sum(dim=1, keepdim=True) + 1e-8)
        # type loss: cross-entropy with soft targets -> implemented as negative sum target * log(pred_prob)
        pred_logprob = F.log_softmax(ty_logits, dim=-1)
        type_loss = - (type_target * pred_logprob).sum(dim=1).mean()
        losses_details["type"] += type_loss.item()

        # regression targets: expected time/mag under P
        # expected_time = (P @ gt_times) scaled by M and normalized by present_target (to handle low assignments)
        gt_times = gt_tensors["gt_times"]
        gt_mags = gt_tensors["gt_mags"]
        denom = row_sum * float(M) + eps

        expected_time = (P @ gt_times) * float(M) / denom    # [N]
        expected_mag = (P @ gt_mags) * float(M) / denom     # [N]

        # regress only where present_target is non-negligible (weight by present_target)
        time_l2 = ((t_pred - expected_time) ** 2) * (present_target_clamped.detach())
        mag_l2 = ((m_pred - expected_mag) ** 2) * (present_target_clamped.detach())

        time_loss = time_l2.sum() / (present_target_clamped.sum() + eps)
        mag_loss = mag_l2.sum() / (present_target_clamped.sum() + eps)

        losses_details["time"] += time_loss.item()
        losses_details["mag"] += mag_loss.item()

        # priority loss: encourage predicted prio to match expected prio under P
        gt_prios = gt_tensors["gt_prios"]
        expected_prio = (P @ gt_prios) * float(M) / denom
        prio_loss = ((prio - expected_prio) ** 2).mean()
        losses_details["prio"] += prio_loss.item()

        # total per-sample loss (weights)
        w_bce = 1.0; w_type = 1.0; w_time = 1.0; w_mag = 10.0; w_prio = 1.0
        sample_loss = w_bce*bce + w_type*type_loss + w_time*time_loss + w_mag*mag_loss + w_prio*prio_loss
        total_loss = total_loss + sample_loss

    total_loss = total_loss / float(B)
    total_loss.backward()
    opt.step()

    # aggregate losses for printing
    for k in losses_details:
        losses_details[k] = losses_details[k] / float(B)
    return total_loss.item(), losses_details, P.detach().cpu()

# ---------------------------
# Run a simple train loop on synthetic data
# ---------------------------
x, gt_list = make_synthetic_batch(B, channels, T, M_max)
x = x.to(device)
print("Sample GT counts:", [len(g) for g in gt_list])

loss, details, lastP = training_step(x, gt_list)
print("Loss:", loss)
print("Loss details:", details)
print("Last transport matrix shape (N x M for sample 0):", lastP.shape)
# Show the soft assignment matrix for sample 0
print("Soft assignment (sample 0):\n", lastP.numpy())

# If you want, run multiple steps to see training behavior
for step in range(3):
    x, gt_list = make_synthetic_batch(B, channels, T, M_max)
    total_loss, d, Pmat = training_step(x.to(device), gt_list)
    print(f"step {step}: loss={total_loss:.4f}, bce={d['bce_present']:.4f}, type={d['type']:.4f}, time={d['time']:.4f}, mag={d['mag']:.4f}")
