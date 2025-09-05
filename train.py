import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from torch_geometric.loader import DataLoader

# -----------------------------
# Training & Evaluation
# -----------------------------
def make_loader(dataset:Dataset, batch_size:int, shuffle:bool, class_balance:bool) -> DataLoader:
    if not class_balance:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Compute weights for graph-level labels
    ys = torch.tensor([int(g.y.item()) for g in dataset.graphs])
    class_counts = torch.bincount(ys)
    weights = 1.0 / (class_counts[ys].float() + 1e-12)
    sampler = WeightedRandomSampler(weights, num_samples=len(ys), replacement=True)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def train_eval(model, train_loader, val_loader, test_loader, device, epochs=200, lr=1e-3, wd=5e-4, patience=50, print_epoch=False):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state = 0.0, None
    bad = 0

    def step(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        L,C,N = 0.0,0,0
        with torch.set_grad_enabled(train):
            for batch in loader:
                batch = batch.to(device)
                out = model(batch)
                # For MultiK models we return (logits, uid). For SingleK, just logits.
                if isinstance(out, tuple):
                    logits, uid = out
                    # Build per-sample targets in the same uid order
                    sid = batch.sample_id.view(-1)
                    ys = batch.y.view(-1)
                    # Map uid -> first y encountered
                    y_s = []
                    for u in uid.tolist():
                        m = (sid == u).nonzero(as_tuple=False).view(-1)
                        y_s.append(int(ys[m[0]].item()))
                    y_target = torch.tensor(y_s, device=device)
                else:
                    logits = out
                    y_target = batch.y.view(-1)
                loss = F.cross_entropy(logits, y_target)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                L += float(loss) * logits.size(0)
                C += int((logits.argmax(-1) == y_target).sum())
                N += logits.size(0)
        return L/max(N,1), C/max(N,1)

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = step(train_loader, True)
        vl, va = step(val_loader, False)
        _, tacc_probe = step(test_loader, False)
        if va > best_val:
            best_val = va; best_state = {k:v.cpu() for k,v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if print_epoch: print(f"Epoch {epoch:03d} | Train loss {tr_loss:.4f} acc {tr_acc:.3f} | Val loss {vl:.4f} acc {va:.3f} | Test acc (probe) {tacc_probe:.3f}")
        if bad >= patience:
            print('Early stopping.'); break

    if best_state is not None:
        model.load_state_dict(best_state)

    def _final(loader):
        _, acc = step(loader, False)
        return acc

    return _final(val_loader), _final(test_loader)