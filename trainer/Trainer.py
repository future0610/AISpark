import torch
import torchvision
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import warnings
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import time

warnings.filterwarnings("ignore")

WORKDIR = os.path.abspath(os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), os.path.pardir))

# 이미지를 보여주기 위한 헬퍼(helper) 함수
# (아래 `plot_classes_preds` 함수에서 사용)
def matplotlib_imshow(img, one_channel = False):
    if one_channel:
        img = img.mean(dim = 0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap = "Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
class Trainer:
    def __init__(self, model, optimizer, loss_func, eval_metric, device, scheduler = None, **trainer_config):
        self.lr = trainer_config.get("lr", 0.01)
        self.n_cls = trainer_config.get("n_cls", 1)
        self.img_size = trainer_config.get("img_size", 256)
        self.EPOCH = trainer_config.get("epoch", 30)
        self.BATCH = trainer_config.get("batch_size", 16)
        self.barLen = trainer_config.get("bar_len", 20)
        self.log_dir = trainer_config.get("log_dir", "runs")
        self.name = trainer_config.get("experiment", "train")
        self.log_term = trainer_config.get("log_term", 10)
        self.model_name = trainer_config.get("model_name", "model")
        self.save_dir = trainer_config.get("save_dir", "checkpoint")
        self.loss_func = loss_func
        self.eval_metric = eval_metric

        self.start_time = 0

        self.log_dir = os.path.abspath(os.path.join(WORKDIR, self.log_dir))

        self.loaded = False

        self.cost = 0
        self.epoch = -1

        self.device = device

        self.model = model
        self.sota = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        try:
            os.mkdir(self.log_dir)
        except:
            pass
        if Path(os.path.join(self.log_dir, self.name + str(1))).exists():
            i = 1
            while True:
                if not Path(os.path.join(self.log_dir, self.name + str(i))).exists():
                    self.name = self.name + str(i)
                    break
                else:
                    i += 1
        else:
            self.name = self.name + str(1)
        self.logger = None
        self.process = {
            "step": "train",
            "train_loss": [],
            "validate_loss": [],
            "validate_IoU": [],
        }

    def log_img(self, trainloader):
        # 임의의 학습 이미지를 가져옵니다
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        # 이미지 그리드를 만듭니다.
        img_grid = torchvision.utils.make_grid(images)

        # 이미지를 보여줍니다.
        matplotlib_imshow(img_grid, one_channel = False)

        # tensorboard에 기록합니다.
        self.logger.add_image(f'Spark_dataset_{self.BATCH}_batch', img_grid)

        self.logger.add_graph(self.model, images.to(self.device))

    def visualize(self, predicted, mask):
        # 이미지 그리드를 만듭니다.
        img_grid = torchvision.utils.make_grid(predicted)
        labels_grid = torchvision.utils.make_grid(mask)

        # 이미지를 보여줍니다.
        matplotlib_imshow(img_grid, one_channel = False)
        matplotlib_imshow(labels_grid, one_channel = True)

        # tensorboard에 기록합니다.
        self.logger.add_image(f'model_output', img_grid)
        self.logger.add_image(f'Mask', labels_grid)

    def train(self, train_loader):
        self.model.train()
        cost = 0
        size = 0
        step = 0
        running_loss = 0.0
        train_size = len(train_loader.dataset)
        for batch_idx, (img, mask) in enumerate(train_loader):
            self.optimizer.zero_grad()

            img, mask = img.to(self.device), mask.to(self.device)
            predicted = self.model(img)

            loss = self.loss_func(predicted, mask)
            # loss = BCE_IoU_Loss(x = predicted, y = mask, weight = 1, threshold = 0.5)
            cost += loss.item()
            size += img.size(0)
            step += img.size(0)

            loss.backward()
            running_loss += loss.item()
            if batch_idx % self.log_term == self.log_term - 1:    # 매 1000 미니배치마다...
                self.process["train_loss"].append(running_loss / step)
                # ...학습 중 손실(running loss)을 기록하고
                self.logger.add_scalar('Loss/Train',
                                running_loss / self.log_term,
                                self.epoch * len(train_loader) + batch_idx)

                running_loss = 0.0
                step = 0

            print(f"\rEPOCH(lr: {self.optimizer.param_groups[0]['lr']:.4f}) {self.epoch + 1:{len(str(self.EPOCH))}d}/{self.EPOCH} {'Train':35s}:[{'=' * int(self.barLen * (size / train_size)) + '>':{self.barLen + 1}s}][{size:{len(str(train_size))}d}/{train_size}({100 * (size / train_size):6.2f}%)] Loss: {cost / (batch_idx + 1):11.4f}({time.time() - self.start_time:10.3f}s)", end = "")

            self.optimizer.step()
        print()

    def validate(self, val_loader):
        size = 0
        avg_perf = 0
        val_size = len(val_loader.dataset)
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (img, mask) in enumerate(val_loader):
                img, mask = img.to(self.device), mask.to(self.device)

                predicted = self.model(img)
                predicted = 1 * (predicted > 0)

                if batch_idx == 0:
                    self.visualize((torch.sigmoid(predicted) > 0.5).clone().detach().cpu(), mask.clone().detach().cpu())

                avgIoU = self.eval_metric(predicted, mask)
                avg_perf += avgIoU
                size += img.size(0)
                print(f"\rEPOCH(lr: {self.optimizer.param_groups[0]['lr']:.4f}) {self.epoch + 1:{len(str(self.EPOCH))}d}/{self.EPOCH} {'Validation':35s}:[{'=' * int(self.barLen * (size / val_size)) + '>':{self.barLen + 1}s}][{size:{len(str(val_size))}d}/{val_size}({100 * (size / val_size):6.2f}%)] Accuracy: {avg_perf / (batch_idx + 1):7.4f}", end = "")

        self.process["validate_IoU"].append(avg_perf / (batch_idx + 1))
        self.logger.add_scalar('Accuracy/Validate',
                        avg_perf / (batch_idx + 1),
                        self.epoch)
        print()

    def test(self, test_loader):
        self.process["step"] = "test"
        size = 0
        avg_perf = 0
        test_size = len(test_loader.dataset)
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (img, mask) in enumerate(test_loader):
                img, mask = img.to(self.device), mask.to(self.device)

                predicted = self.model(img)
                predicted = 1 * (predicted > 0)

                avgIoU = self.eval_metric(predicted, mask)
                avg_perf += avgIoU
                size += img.size(0)
                print(f"\r{'Test':35s}:[{'=' * int(self.barLen * (size / test_size)) + '>':{self.barLen + 1}s}][{size:{len(str(test_size))}d}/{test_size}({100 * (size / test_size):6.2f}%)] Accuracy: {avg_perf / (batch_idx + 1):7.4f}", end = "")
        print()
        self.logger.close()

    def run(self, train_loader, val_loader = None):
        if self.process["step"] == "test":
            return
        if not self.loaded:
            self.logger = SummaryWriter(log_dir = os.path.join(self.log_dir, self.name))
            self.log_img(train_loader)
            print("Train from scratch.")
            BEST = float("-inf")
            self.start_time = time.time()
        else:
            self.start_time = time.time()
            self.logger = SummaryWriter(log_dir = os.path.join(self.log_dir, self.name))
            self.log_img(train_loader)

            for e in range(self.epoch + 1):
                self.logger.add_scalar("Learning Rate",
                                    self.scheduler.lr_lambdas[0](e),
                                    e)
            for i in range(len(self.process["train_loss"])):
                # ...학습 중 손실(running loss)을 기록하고
                self.logger.add_scalar('Loss/Train',
                                self.process["train_loss"][i],
                                self.epoch * len(train_loader) + i)
            if val_loader != None:
                for i in range(len(self.process["validate_loss"])):
                    self.logger.add_scalar('Loss/Validate',
                                    self.process["validate_loss"][i],
                                    self.epoch * len(val_loader) + i)

                for i in range(len(self.process["validate_IoU"])):
                    self.logger.add_scalar('Accuracy/Validate',
                                    self.process["validate_IoU"][i],
                                    i)
                
                BEST = max(self.process["validate_IoU"])
                if self.process["step"] == "train":
                    self.validate(val_loader)
                    if self.process["validate_IoU"][-1] >= BEST:
                        BEST = self.process["validate_IoU"][-1]
                        self.save(self.model, self.optimizer, exp_name = self.name, fileName = self.model_name + "_SOTA")
        
        print("Train Start...")
        for epoch in range(self.epoch + 1, self.EPOCH):
            self.epoch = epoch
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.add_scalar("Learning Rate",
                                lr,
                                epoch)
            self.process["step"] = "train"
            self.train(train_loader)
            self.save(self.model, self.optimizer, exp_name = self.name, fileName = self.model_name)
            print("Saved.")
            self.process["step"] = "validate"
            if val_loader != None:
                self.validate(val_loader)
                if self.process["validate_IoU"][-1] > BEST:
                    BEST = self.process["validate_IoU"][-1]
                    self.save(self.model, self.optimizer, exp_name = self.name, fileName = self.model_name + "_SOTA")
                    print("SOTA Updated.")
            self.scheduler.step()
            self.save(self.model, self.optimizer, exp_name = self.name, fileName = self.model_name)
            print()            

    def save(self, net: nn.Module, optimizer: torch.optim.Optimizer, exp_name = "./save/", fileName = "model"):
        exp_dir = os.path.abspath(os.path.join(self.log_dir, exp_name))
        exp_dir = os.path.abspath(os.path.join(exp_dir, self.save_dir))
        root = exp_dir
        if not self.loaded and not Path(root).exists():
            os.mkdir(root)

        torch.save({
                'epoch': self.epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': self.process['train_loss'],
                'validate_loss': self.process['validate_loss'],
                'validate_IoU': self.process['validate_IoU'],
                "step": self.process["step"]
                }, os.path.abspath(os.path.join(root, fileName + ".pth")))

    def deploy(self, root, fileName):
        root = os.path.abspath(os.path.join(root, fileName + ".pth"))
        
        if not self.loaded and not Path(root).exists():
            os.mkdir(root)

        torch.save(self.model.state_dict(), root)

    def load(self, exp_name = None, fileName = None):
        if exp_name == None:
            exp_name = self.name
        else:
            self.name = exp_name
        if fileName == None:
            fileName = self.model_name
        print("Loading process...")
        try:
            exp_dir = os.path.abspath(os.path.join(self.log_dir, exp_name))
            for p in Path(exp_dir).glob("*.tfevents.*"):
                p.unlink()
            exp_dir = os.path.abspath(os.path.join(exp_dir, self.save_dir))
            checkpoint = torch.load(os.path.abspath(os.path.join(exp_dir, fileName + ".pth")))
        except RuntimeError as e:
            print(e)
            print(f"RuntimError: {e}")
            print()
            return

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.epoch = checkpoint['epoch']
        self.process['train_loss'] = checkpoint['train_loss']
        self.process['validate_loss'] = checkpoint['validate_loss']
        self.process['validate_IoU'] = checkpoint['validate_IoU']
        self.process["step"] = checkpoint["step"]
        
        if self.process["step"] == "validate":
            self.loaded  = True
            print("Loaded.")
            return
        self.scheduler.last_epoch = self.epoch
        self.scheduler.step()
        self.loaded = True
        print("Loaded.")

    def get_model(self, root, fileName):
        try:
            root = os.path.abspath(os.path.join(root, fileName + "pth"))
            checkpoint = torch.load(root)
        except FileNotFoundError as f:
            print(f"File does not exist.")
        except RuntimeError as r:
            print(f"RuntimeError: {r}")
        finally:
            pass
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print("done")