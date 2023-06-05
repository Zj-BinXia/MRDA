import os
import utility2
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util
from utils import util2
from collections import OrderedDict
import random
import numpy as np
import torch.nn as nn


class Trainer():
    def __init__(self, args, loader, model_meta_copy, model_TA, my_loss, ckp):
        self.test_res_psnr = []
        self.test_res_ssim = []
        self.args = args
        self.scale = args.scale
        self.loss1= nn.L1Loss()
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model_meta_copy = model_meta_copy

        self.model_meta_copy_state = self.model_meta_copy.state_dict()
        self.model_TA = model_TA
        self.loss = my_loss
        self.meta_batch_size = args.meta_batch_size
        self.task_batch_size = args.task_batch_size
        self.task_iter = args.task_iter
        self.test_iter = args.test_iter
        self.optimizer = utility2.make_optimizer(args, self.model_TA)
        self.scheduler = utility2.make_scheduler(args, self.optimizer)
        self.lr_task = args.lr_task
        self.taskiter_weight=[1/self.task_iter]*self.task_iter


        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model_TA.train()

        degrade = util2.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

        timer = utility2.timer()

        for batch, (hr, _) in enumerate(self.loader_train):
            hr = hr.cuda() # b, c, h, w
            timer.tic()
            loss_all = 0

            lr_blur, hr_blur = degrade(hr)  # b, c, h, w
            self.model_meta_copy.get_model().load_state_dict(self.model_meta_copy_state)

            for iter in range(self.test_iter):
                if iter == 0:
                    learning_rate = 1e-2
                elif iter < 5:
                    learning_rate = 5e-3
                else:
                    learning_rate = 1e-3
                sr,_ = self.model_meta_copy(lr_blur, hr_blur)
                loss = self.loss1(sr, hr)
                self.model_meta_copy.zero_grad()
                loss.backward()

                for param in self.model_meta_copy.parameters():
                    if param.requires_grad:
                        param.data.sub_(param.grad.data * learning_rate)

            _, deg_repre = self.model_meta_copy(lr_blur, hr_blur)
            sr,_ = self.model_TA(lr_blur, deg_repre.detach())
            loss_all += self.loss1(sr,hr)
            self.optimizer.zero_grad()
            loss_all.backward()
            self.optimizer.step()

            # Remove the hooks before next training phase
            timer.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                    'Loss [SR loss:{:.3f}]\t'
                    'Time [{:.1f}s]'.format(
                        epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                        loss_all.item(),
                        timer.release(),
                    ))

        self.loss.end_log(len(self.loader_train))

        # save model
        if epoch > self.args.st_save_epoch or (epoch %20 ==0):
            target = self.model_TA.get_model()
            model_dict = target.state_dict()
            torch.save(
                model_dict,
                os.path.join(self.ckp.dir, 'model', 'model_TA_{}.pt'.format(epoch))
            )

            optimzer_dict = self.optimizer.state_dict()
            torch.save(
                optimzer_dict,
                os.path.join(self.ckp.dir, 'optimzer', 'optimzer_TA_{}.pt'.format(epoch))
            )

        target = self.model_TA.get_model()
        model_dict = target.state_dict()
        torch.save(
            model_dict,
            os.path.join(self.ckp.dir, 'model', 'model_TA_last.pt')
        )
        optimzer_dict = self.optimizer.state_dict()
        torch.save(
            optimzer_dict,
            os.path.join(self.ckp.dir, 'optimzer', 'optimzer_TA_last.pt')
        )


    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))

        timer_test = utility2.timer()
        t = 0
        self.model_TA.eval()

        degrade = util2.SRMDPreprocessing(
                self.scale[0],
                kernel_size=self.args.blur_kernel,
                blur_type=self.args.blur_type,
                sig=self.args.sig,
                lambda_1=self.args.lambda_1,
                lambda_2=self.args.lambda_2,
                theta=self.args.theta,
                noise=self.args.noise
            )

        for idx_scale, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0
                iter_psnr = [0] * (self.test_iter + 1)
                iter_ssim = [0] * (self.test_iter + 1)
                for idx_img, (hr, filename) in enumerate(d):
                    self.model_meta_copy.get_model().load_state_dict(self.model_meta_copy_state)
                    hr = hr.cuda()  # b, c, h, w
                    hr = self.crop_border(hr, scale)
                    # inference
                    # timer_test.tic()
                    lr_blur, hr_blur = degrade(hr, random=False)



                    for iter in range(self.test_iter):
                        if iter == 0:
                            learning_rate = 1e-2
                        elif iter < 5:
                            learning_rate = 5e-3
                        else:
                            learning_rate = 1e-3

                        sr,_ = self.model_meta_copy(lr_blur, hr_blur)
                        loss = self.loss1(sr, hr)
                        self.model_meta_copy.zero_grad()
                        loss.backward()

                        for param in self.model_meta_copy.parameters():
                            if param.requires_grad:
                                param.data.sub_(param.grad.data * learning_rate)

                    _, deg_repre = self.model_meta_copy(lr_blur, hr_blur)
                    #print(lr_blur.shape,deg_repre.shape)
                    torch.cuda.synchronize()
                    timer_test.tic()
                    sr = self.model_TA(lr_blur, deg_repre.detach())
                    torch.cuda.synchronize()
                    timer_test.hold()
                    t0 = timer_test.release()

                    print("idx:", idx_img, ",time consuming:", t0)
                    t += t0

                    hr = utility2.quantize(hr, self.args.rgb_range)
                    sr = utility2.quantize(sr, self.args.rgb_range)

                    iter_psnr[-1] += utility2.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=True
                    )
                    iter_ssim[-1] += utility2.calc_ssim(
                        (sr * 255).round().clamp(0, 255), (hr * 255).round().clamp(0, 255), scale,
                        benchmark=True
                    )

                    timer_test.hold()

                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        self.ckp.save_results(filename, save_list, scale)

            # for t in range(self.test_iter+1):
            #     print("iter:",t,",PSNR:",iter_psnr[t]/ len(self.loader_test),",SSIM:",iter_ssim[t]/ len(self.loader_test))

            eval_psnr = iter_psnr[-1]
            eval_ssim = iter_ssim[-1]

            if len(self.test_res_psnr)>10:
                self.test_res_psnr.pop(0)
                self.test_res_ssim.pop(0)
            self.test_res_psnr.append(eval_psnr / len(self.loader_test))
            self.test_res_ssim.append(eval_ssim / len(self.loader_test))
            print("All time consuming:", t / len(self.loader_test))

            self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
            self.ckp.write_log(
                '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} mean_PSNR: {:.3f} mean_SSIM: {:.4f}'.format(
                    self.args.resume,
                    self.args.data_test,
                    scale,
                    eval_psnr / len(self.loader_test),
                    eval_ssim / len(self.loader_test),
                    np.mean(self.test_res_psnr),
                    np.mean(self.test_res_ssim)
                ))

    def crop_border(self, img_hr, scale):
        b, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr

    def get_patch(self, img, patch_size=48, scale=4):
        tb, tc, th, tw = img.shape  ## HR image
        tp = round(scale * patch_size)
        tx = random.randrange(0, (tw - tp))
        ty = random.randrange(0, (th - tp))

        return img[:,:,ty:ty + tp, tx:tx + tp]

    def crop(self, img_hr):
        # b, c, h, w = img_hr.size()
        tp_hr = []
        for i in range(self.task_batch_size):
            tp_hr.append(self.get_patch(img_hr,self.args.patch_size,self.scale[0]))
        tp_hr = torch.cat(tp_hr,dim=0)
        return tp_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >=  self.args.epochs_sr