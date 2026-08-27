import os, torch
# import shutil
import numpy as np
# from torch import nn
# from torch.optim import lr_scheduler as lrs
import time
# import eikonal
import tifffile as tiff
# from src.cellpose_omni import plot


class TakeNotesLoss:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.id = -1

    def update(self, value):
        self.sum += value
        self.count += 1

    def update2(self):
        tmp = self.sum / self.count
        self.sum = 0
        self.count = 0
        self.id += 1
        return tmp


class Trainer:
    def __init__(self, data_loader, test_loader, model, loss_criterion, optimizer, lr_scheduler, eval_metric, eval_PR, rootPath=None, modelPath=None, device=torch.device('cpu'), batchSize=1):
        self.rootPath = rootPath
        self.batchSize = batchSize
        self.dataLoader = data_loader
        self.testLoader = test_loader
        self.device = device
        # self.valCount = int(30 / self.batchSize)
        self.valCount = 50
        # self.valCount = 150
        # Network
        self.super_net = model
        self.super_net.to(self.device)
        '''加载损失函数优化器学习率等'''
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_metric = eval_metric
        self.eval_PR = eval_PR
        self.modelPath = modelPath
        if self.modelPath is None:
            self.modelPath = './saved_models/'
        if not os.path.isdir(self.modelPath):
            os.makedirs(self.modelPath)
        if self.rootPath is None:
            self.rootPath = './'
        self.text_eval_path = os.path.join(self.modelPath, 'text_eval.txt')
        self.text_save_model_path = os.path.join(self.modelPath, 'text_save_model.txt')
        self.text_loss_path = os.path.join(self.modelPath, 'text_loss.txt')
        self.text_precision_path = os.path.join(self.modelPath, 'text_precision.txt')
        self.text_recall_path = os.path.join(self.modelPath, 'text_recall.txt')
        self.text_f1_path = os.path.join(self.modelPath, 'text_f1.txt')
        with open(self.text_eval_path, 'w', encoding='utf-8') as f:
            pass
        with open(self.text_save_model_path, 'w', encoding='utf-8') as f:
            pass
        with open(self.text_loss_path, 'w', encoding='utf-8') as f:
            pass
        with open(self.text_precision_path, 'w', encoding='utf-8') as f:
            pass
        with open(self.text_recall_path, 'w', encoding='utf-8') as f:
            pass
        with open(self.text_f1_path, 'w', encoding='utf-8') as f:
            pass
        self.stop_training = False  # 添加一个标志变量，用于控制是否停止训练
        self.tt2 = 0
        self.modSavePath = ""

    def Train(self, turn=2, writer=None):
        try:
            train_small_losses = TakeNotesLoss()
            train_big_losses = TakeNotesLoss()
            evalVal = TakeNotesLoss()
            lastEvalVal = 0
            self.valCount = min(len(self.dataLoader) - 1, self.valCount)
            # self.valCount = 1
            iter_count = 0
            # sigmod = nn.Sigmoid()
            for t in range(turn):
                if self.stop_training:  # 检查是否需要停止训练
                    print("训练已停止")
                    break

                torch.cuda.empty_cache()
                torch.set_grad_enabled(True)
                self.super_net.train()
                start = time.time()
                for kk, (img, mask, dist, name) in enumerate(self.dataLoader):
                    if self.stop_training:  # 检查是否需要停止训练
                        print("训练已停止")
                        break

                    if img.shape[0] != self.batchSize:
                        continue
                    torch.cuda.empty_cache()
                    img = img.to(self.device)
                    mask = mask.to(self.device)
                    dist = dist.to(self.device)

                    # with torch.no_grad():
                    #     # 计算欧几里得距离场
                    #     distance = eikonal.solve_eikonal(mask, eps=1e-5, min_steps=200, use_triton=False)
                    #     gradient = eikonal.gradient_from_eikonal(distance)
                    # seg = self.super_net(img, dist)
                    seg = self.super_net(dist)
                    # seg = net_res[:, 0:1]
                    # gra = net_res[:, 1:]
                    if (iter_count + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            ns = seg.cpu().detach().numpy()  # 训练结果--Tag
                            ma = mask.cpu().detach().numpy()  # 目标标签
                            dt = dist.cpu().detach().numpy()  # 目标距离场
                            imgs = img.cpu().detach().numpy()  # 原图
                            # ngi = gra.cpu().detach().numpy()  # 训练结果--Morphological gradient
                            # gri = gradient.cpu().detach().numpy()  # 目标梯度
                            nsi = ns[:, 0]
                            mai = ma[:, 0]
                            dti = dt[:, 0]
                            imgi = imgs[:, 0]
                            # ngi = np.transpose(ng, (0, 2, 3, 4, 1))
                            # gri = np.transpose(gr, (0, 2, 3, 4, 1))
                            self.save_result(nsi, mai, dti, imgi)
                        torch.cuda.empty_cache()

                    # loss = self.loss_criterion(seg, mask, gra, gradient)[0]
                    loss = self.loss_criterion(seg, mask)[0]
                    train_small_losses.update(loss.item())
                    train_big_losses.update(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if (iter_count + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            p, r, _ = self.eval_PR(seg, mask)
                            f1 = 2 * p * r / (p + r)
                            print("precision: {:.4f}, recall: {:.4f}, f1-score: {:.4f}".format(p, r, f1))
                            with open(self.text_precision_path, "a", encoding="utf-8") as f:
                                f.write(f"{iter_count + 1} {p:.4f}\n")
                            f.close()
                            with open(self.text_recall_path, "a", encoding="utf-8") as f:
                                f.write(f"{iter_count + 1} {r:.4f}\n")
                            f.close()
                            with open(self.text_f1_path, "a", encoding="utf-8") as f:
                                f.write(f"{iter_count + 1} {f1:.4f}\n")
                            f.close()
                        torch.cuda.empty_cache()

                    if (iter_count + 1) % 10 == 0:
                        tmpLoss = train_small_losses.update2()
                        writer.add_scalar('Loss/TrainSmallLoss', tmpLoss, train_small_losses.id)
                        print('TRAIN [Epoch %d | %d] [Proce %d | %d] [Loss %.4f] [%d]' % (t, turn, kk, len(self.dataLoader),
                                                                                          tmpLoss, iter_count + 1))
                        with open(self.text_loss_path, "a", encoding="utf-8") as f:
                            f.write(f"{iter_count + 1} {tmpLoss:.4f}\n")
                        f.close()
                    if (iter_count + 1) % self.valCount == 0:
                        torch.cuda.empty_cache()
                        writer.add_scalar('Loss/TrainBigLoss', train_big_losses.update2(), train_big_losses.id)
                        self.super_net.eval()
                        with torch.no_grad():
                            for kk, (img, mask, dist, name) in enumerate(self.testLoader):
                                if self.stop_training:  # 检查是否需要停止训练
                                    print("训练已停止")
                                    break

                                if img.shape[0] != self.batchSize:
                                    continue
                                # img = img.to(self.device)
                                mask = mask.to(self.device)
                                dist = dist.to(self.device)

                                # # 计算欧几里得距离场
                                # distance = eikonal.solve_eikonal(mask, eps=1e-5, min_steps=200, use_triton=False)
                                # gradient = eikonal.gradient_from_eikonal(distance)

                                # seg = self.super_net(img, dist)
                                seg = self.super_net(dist)
                                # seg = net_res[:, 0:1]
                                # gra = net_res[:, 1:]
                                # loss0 = self.loss_criterion(seg0, mask) * 0.5
                                # loss1 = self.loss_criterion(seg1, smallMask) * 0.5
                                # loss = loss0 + loss1
                                # valLoss.update(loss.item())
                                # valLoss1.update(loss0.item())
                                # valLoss2.update(loss1.item())
                                # seg = sigmod(seg)
                                # eval = self.eval_metric(seg, mask, gra, gradient)
                                eval = self.eval_metric(seg, mask)
                                evalVal.update(eval)
                            curEvalVal = evalVal.update2()
                            writer.add_scalar('Eval/EvalVal', curEvalVal, evalVal.id)
                            curEvalVal = curEvalVal
                            # if (curEvalVal > lastEvalVal or abs(curEvalVal - lastEvalVal) <= 0.001) and t > 10:
                            if curEvalVal > lastEvalVal and t > 10:
                                print('验证集指标改善，保存模型')
                                self.modSavePath = os.path.abspath(os.path.join(self.modelPath, "supernet_000.pth"))
                                # torch.save({'state_dict': self.super_net.state_dict(), 'param': self.optimizer}, os.path.join(self.modelPath, "supernet_%s.pth" % (str(t).zfill(5))))
                                torch.save({'state_dict': self.super_net.state_dict(), 'param': self.optimizer},
                                           self.modSavePath)

                                lastEvalVal = curEvalVal
                                with open(self.text_save_model_path, "a", encoding="utf-8") as f:
                                    f.write(f"{iter_count + 1} 验证集损失减少,保存模型 {curEvalVal:.4f}\n")
                                f.close()
                            # if (t + 1) % 20 == 0 and t < 100:
                            #     torch.save({'state_dict': self.super_net.state_dict(), 'param': self.optimizer},
                            #                os.path.join(self.modelPath, "supernet_%s.pth" % (str(t).zfill(5))))
                            self.lr_scheduler.step(curEvalVal)
                            lr = self.optimizer.param_groups[0]['lr']
                            writer.add_scalar('TrainParam/Lr', lr, evalVal.id)
                            print('VAL [Epoch %d | %d] [EvalVal; %.4f] [Lr: %.8f] [%d]' % (t, turn, curEvalVal, lr,
                                                                                         iter_count + 1))
                            with open(self.text_eval_path, "a", encoding="utf-8") as f:
                                f.write(f"{iter_count + 1} {curEvalVal:.4f}\n")
                            f.close()
                        torch.cuda.empty_cache()
                        self.super_net.train()
                    iter_count += 1
                    # print("iter_count：", iter_count)
                    # if kk > 100:
                    #     break
                print("用时：", time.time() - start)
                print("")
                self.tt2 = t + 1
            time.sleep(1)
            del self.dataLoader
            del self.testLoader
            del self.loss_criterion
            del self.optimizer
            del self.lr_scheduler
            del self.eval_metric
            # 删除模型引用
            if hasattr(self, 'super_net'):
                del self.super_net
                print("模型已删除")
            # 清理 GPU Cache
            torch.cuda.empty_cache()
            print("GPU 缓存已清理！")
        except Exception as e:
            print("结束")
        # torch.save({'state_dict': self.super_net.state_dict(), 'param': self.optimizer},
        #            os.path.join(self.modelPath, "supernet_001.pth"))

    def save_result(self, net_seg, mask, dist, img):
        # nclasses = 3
        for i in range(len(mask)):
            root = fr"{self.rootPath}\train_results2\{i + 1}"
            os.makedirs(root, exist_ok=True)
            net_segi = ((net_seg[i] - net_seg[i].min()) / (net_seg[i].max() - net_seg[i].min()) * 255).astype(np.uint8)
            maski = ((mask[i] - mask[i].min()) / (mask[i].max() - mask[i].min()) * 255).astype(np.uint8)
            disti = ((dist[i] - dist[i].min()) / (dist[i].max() - dist[i].min()) * 255).astype(np.uint8)
            imgi = ((img[i] - img[i].min()) / (img[i].max() - img[i].min()) * 255).astype(np.uint8)
            tiff.imwrite(os.path.join(root, f"net_seg{i + 1}.tif"), net_segi)
            tiff.imwrite(os.path.join(root, f"mask{i + 1}.tif"), maski)
            tiff.imwrite(os.path.join(root, f"dist{i + 1}.tif"), disti)
            tiff.imwrite(os.path.join(root, f"img{i + 1}.tif"), imgi)

            # tiff.imwrite(os.path.join(root, f"net_grad{i + 1}.tif"), np.transpose(net_grad[i], (1, 2, 3, 0)))
            # tiff.imwrite(os.path.join(root, f"net_grad_color{i + 1}.tif"),
            #              plot.dx_to_circ(net_grad[i], transparency=True)
            #              if nclasses > 1 else np.zeros(
            #                  np.expand_dims(net_segi, axis=0).shape + (3 + True,), np.uint8)
            #              )
            # tiff.imwrite(os.path.join(root, f"grad{i + 1}.tif"), np.transpose(grad[i], (1, 2, 3, 0)))
            # tiff.imwrite(os.path.join(root, f"grad_color{i + 1}.tif"),
            #              plot.dx_to_circ(grad[i], transparency=True)
            #              if nclasses > 1 else np.zeros(
            #                  np.expand_dims(maski, axis=0).shape + (3 + True,), np.uint8)
            #              )

    def Stop2(self):
        self.stop_training = True
        time.sleep(3)
        try:
            # 删除模型引用
            if hasattr(self, 'super_net'):
                del self.super_net
                print("模型已删除")
            # 清理 GPU Cache
            torch.cuda.empty_cache()
            print("GPU 缓存已清理")
            del self.dataLoader
            del self.testLoader
            del self.loss_criterion
            del self.optimizer
            del self.lr_scheduler
            del self.eval_metric
        except Exception as e:
            print("Stop")

