import sys
sys.path.append('.')

import os
import torch
from sklearn.metrics import classification_report
from trainer.substra.algo import Algo
from common.utils import Mapping
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType
)

DEVICE = "cpu"


class MonaiAlgo(Algo):
    def __init__(self):
        self.model = None
        self.loss = None
        self.optimizer = None
        self.epochs = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.act = None
        self.to_onehot = None
        self.model_dir = None
        self.dice_metric = None

    def train(self, X, y, models, rank):
        epoch_num = 4

    def train(self):
        # """## Set deterministic training for reproducibility"""
        set_determinism(seed=0)
        device = torch.device(DEVICE)
        val_interval = 2
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []

        post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=2)])
        post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=2)])


        self.model.to(device)
        for epoch in range(self.epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.epochs}")
            self.model.train()
            epoch_loss = 0
            step = 0
            for batch_data in self.train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                print(
                    f"{step}/{len(self.train_ds) // self.train_loader.batch_size}, "
                    f"train_loss: {loss.item():.4f}")
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    for val_data in self.val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        roi_size = (160, 160, 160)
                        sw_batch_size = 2
                        val_outputs = sliding_window_inference(
                            val_inputs, roi_size, sw_batch_size, self.model)
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                        # compute metric for current iteration
                        self.dice_metric(y_pred=val_outputs, y=val_labels)

                    # aggregate the final mean dice result
                    metric = self.dice_metric.aggregate().item()
                    # reset the status for next validation round
                    self.dice_metric.reset()
                    # best_model = self.model
                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        best_model = self.model.state_dict()
                        # torch.save(model.state_dict(), os.path.join(
                        #     root_dir, "best_metric_model.pth"))
                        # print("saved new best metric model")

                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )
        checkpoint = Mapping()
        # np_arr = torch_tensor
        checkpoint.update(epoch=best_metric_epoch, weights=best_model, metric=best_metric)
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        return checkpoint


    def load_model(self, client):
        path = client.modelFile
        self.model.load_state_dict(torch.load(path))
        print("model loaded and creating report...")

    def save_model(self, model, path):
        pass
        # json.dump(model, path)

    def predict(self, client, model, class_names):
        set_determinism(seed=0)
        device = torch.device(DEVICE) 
        self.load_model(client)
        self.model.to(device)
        self.model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            for test_data in self.test_loader:
                test_images, test_labels = (
                    test_data[0].to(device),
                    test_data[1].to(device),
                )
                pred = model(test_images).argmax(dim=1)
                for i in range(len(pred)):
                    y_true.append(test_labels[i].item())
                    y_pred.append(pred[i].item())

        test_report = Mapping()
        test_report.update(report=classification_report(
        y_true, y_pred, target_names=class_names, digits=4))

        return test_report 