
import torch
import torch.nn.functional as F
import math
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from collections import defaultdict
import copy
from uni_func import UniversalClassifier




def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }
    


def print_model_mib(model, name="Model", include_buffers=True):
    """
    Print model parameter count and size in MiB.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    buffer_bytes = 0
    if include_buffers:
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    total_bytes = param_bytes + buffer_bytes
    total_mib = total_bytes / (1024 ** 2)

    param_mib = param_bytes / (1024 ** 2)
    buffer_mib = buffer_bytes / (1024 ** 2)

    print(f"=== {name} ===")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Param size : {param_mib:.3f} MiB")
    if include_buffers:
        print(f"Buffer size: {buffer_mib:.3f} MiB")
    print(f"Total size : {total_mib:.3f} MiB")

def print_param_table(model):
    total = 0
    print("=" * 80)
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            size_mb = params * 4 / 1024 / 1024
            print(f"{name:<40} {params:>10} params | {size_mb:>6.3f} MB")
            total += params
    print("=" * 80)
    print(f"TOTAL: {total} params | {total*4/1024/1024:.3f} MB")


class ModelTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        task_type="classification",
        classifier=None,
        vocab_size=None,
    ):
        self.model = model
        self.classifier = classifier
        self.optimizer = optimizer
        self.device = device
        self.task_type = task_type
        self.vocab_size = vocab_size

    def train_epoch(self, loader):
        self.model.train()
        if self.classifier is not None:
            self.classifier.train()

        total_loss = 0.0
        total, correct = 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            # ---- Classification ----
            if self.task_type == "classification":
                feats = self.model(x)
                logits = self.classifier(feats)
                loss = F.cross_entropy(logits, y)

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            # ---- Language Model ----
            elif self.task_type == "lm":
                logits = self.model(x)  # (B, T, V)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, self.vocab_size),
                    x[:, 1:].reshape(-1)
                )

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if self.task_type == "classification":
            return total_loss / len(loader), correct / total
        else:
            return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        if self.classifier is not None:
            self.classifier.eval()

        total_loss = 0.0
        total, correct = 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            if self.task_type == "classification":
                feats = self.model(x)
                logits = self.classifier(feats)
                loss = F.cross_entropy(logits, y)

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            elif self.task_type == "lm":
                logits = self.model(x)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, self.vocab_size),
                    x[:, 1:].reshape(-1)
                )

            total_loss += loss.item()

        if self.task_type == "classification":
            return total_loss / len(loader), correct / total
        else:
            avg_loss = total_loss / len(loader)
            ppl = math.exp(avg_loss)
            return avg_loss, ppl

class ExperimentRunner:
    def __init__(
        self,
        models,
        dataset,
        num_folds,
        num_epochs,
        batch_size,
        lr,
        task_type="classification",
        classifier_in=None,
        vocab_size=None,
    ):
        self.models = models
        self.dataset = dataset
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.task_type = task_type
        self.classifier_in = classifier_in
        self.vocab_size = vocab_size

        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
         #torch.device( "mps" if torch.backends.mps.is_available() else "cpu"

        

        #
        self.logs = defaultdict(list)

    def run(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        for name, model_instance in self.models.items():
            print(f"\n>>> Model: {name}")

            for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
                print(f"\n--- Fold {fold} ---")

                train_loader = DataLoader(
                    Subset(self.dataset, train_idx),
                    batch_size=self.batch_size,
                    shuffle=True,
                )
                val_loader = DataLoader(
                    Subset(self.dataset, val_idx),
                    batch_size=self.batch_size,
                    shuffle=False,
                )

                model = copy.deepcopy(model_instance).to(self.device)

                # -------------------------
                # classifier (only for classification)
                # -------------------------
                classifier = None
                params = list(model.parameters())

                if self.task_type == "classification":
                    classifier = UniversalClassifier(
                        self.classifier_in, d_out=10
                    ).to(self.device)
                    params += list(classifier.parameters())

                optimizer = torch.optim.AdamW(params, lr=self.lr)

                trainer = ModelTrainer(
                    model=model,
                    classifier=classifier,
                    optimizer=optimizer,
                    device=self.device,
                    task_type=self.task_type,
                    vocab_size=self.vocab_size,
                )

                # -------------------------
                # training loop
                # -------------------------
                for epoch in range(1, self.num_epochs + 1):

                    if self.task_type == "classification":
                      
                        train_loss, train_acc = trainer.train_epoch(train_loader)
                        val_loss, val_acc = trainer.eval_epoch(val_loader)

                        # ---- log ----
                        self.logs["records"].append({
                            "model": name,
                            "fold": fold,
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "train_acc": train_acc,
                            "val_acc": val_acc,
                        })

                        print(
                            f"{name} | Fold {fold} | Epoch {epoch} "
                            f"| TrainAcc={train_acc:.4f} "
                            f"| ValAcc={val_acc:.4f}"
                        )

                    else:
                        # LM / regression
                        train_loss = trainer.train_epoch(train_loader)
                        val_loss, ppl = trainer.eval_epoch(val_loader)

                        # ---- log ----
                        self.logs["records"].append({
                            "model": name,
                            "fold": fold,
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "ppl": ppl,
                        })

                        print(
                            f"{name} | Fold {fold} | Epoch {epoch} "
                            f"| TrainLoss={train_loss:.4f} "
                            f"| ValLoss={val_loss:.4f} "
                            f"| PPL={ppl:.2f}"
                        )

        return self.logs
    

def save_experiment_csv(logs, path="experiment_logs.csv"):
    df = pd.DataFrame(logs["records"])
    df.to_csv(path, index=False)
    print(f"📁 Saved experiment results to: {path}")




class LMTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        vocab_size,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.vocab_size = vocab_size

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for x in loader:
            # x: (B, T)
            x = x.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(x)  # (B, T, V)

            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                x[:, 1:].reshape(-1)
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0

        for x in loader:
            x = x.to(self.device)

            logits = self.model(x)

            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                x[:, 1:].reshape(-1)
            )

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        ppl = math.exp(avg_loss)

        return avg_loss, ppl


class LMExperimentRunner:
    def __init__(
        self,
        models: dict,
        train_dataset,
        val_dataset,
        num_epochs: int,
        batch_size: int,
        lr: float,
        vocab_size: int,
    ):
        self.models = models
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.vocab_size = vocab_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logs = defaultdict(list)

    def run(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        for name, model_instance in self.models.items():
            print(f"\n>>> Model: {name}")

            model = copy.deepcopy(model_instance).to(self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr
            )

            trainer = LMTrainer(
                model=model,
                optimizer=optimizer,
                device=self.device,
                vocab_size=self.vocab_size,
            )

            for epoch in range(1, self.num_epochs + 1):
                train_loss = trainer.train_epoch(train_loader)
                val_loss, ppl = trainer.eval_epoch(val_loader)

                self.logs["records"].append({
                    "model": name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "ppl": ppl,
                })

                print(
                    f"{name} | Epoch {epoch} "
                    f"| TrainLoss={train_loss:.4f} "
                    f"| ValLoss={val_loss:.4f} "
                    f"| PPL={ppl:.2f}"
                )

        return self.logs
