import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import AttnLSTM

class NpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.long)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="weights.pt")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "X.npy")  # (N,30,822)
    y = np.load(data_dir / "y.npy")  # (N,)

    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * 0.9)
    tr, va = idx[:split], idx[split:]

    train_ds = NpyDataset(X[tr], y[tr])
    valid_ds = NpyDataset(X[va], y[va])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=False)

    classes = int(y.max()) + 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttnLSTM(classes=classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, ok, loss_sum = 0, 0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            loss_sum += float(loss) * len(yb)
            pred = logits.argmax(dim=1)
            ok += int((pred == yb).sum())
            tot += len(yb)

        model.eval()
        vok, vtot = 0, 0
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                vok += int((pred == yb).sum())
                vtot += len(yb)

        tracc = ok / max(1, tot)
        vacc = vok / max(1, vtot)
        print(f"ep {ep:02d} loss {loss_sum/max(1,tot):.4f} train {tracc:.3f} valid {vacc:.3f}")

        if vacc > best:
            best = vacc
            torch.save({"model": model.state_dict(), "classes": classes}, args.out)
            print("saved:", args.out)

    print("best valid:", best)

if __name__ == "__main__":
    main()
