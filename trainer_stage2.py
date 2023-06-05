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
    def __init__(self, args, loader, model_meta, model_meta_copy, my_loss, ckp):
        self.test_res_psnr = []
        self.test_res_ssim = []
        self.args = args
        self.scale = args.scale
        self.loss1= nn.L1Loss()
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model_meta = model_meta
        self.model_meta_copy = model_meta_copy

        self.loss = my_loss
        self.meta_batch_size = args.meta_batch_size
        self.task_batch_size = args.task_batch_size
        self.task_iter = args.task_iter
        self.test_iter = args.test_iter
        self.optimizer = utility2.make_optimizer(args, self.model_meta)
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
        self.model_meta.train()

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
        losses_sr = utility2.AverageMeter()

        for batch, (hr, _,) in enumerate(self.loader_train):
            hr = hr.cuda() # b, c, h, w
            b,c,h,w =hr.shape
            query_label = hr[:b//2,:,:,:]
            support_label = hr[b//2:,:,:,:]
            weights = OrderedDict(
                (name, param) for (name, param) in self.model_meta.model.named_parameters() if param.requires_grad )

            meta_grads = {name: 0 for (name, param) in weights.items()}
            timer.tic()
            loss_train = [0]*self.task_iter
            for i in range(self.meta_batch_size):
                lr_blur, hr_blur = degrade(hr)  # b, c, h, w
                query_lr = lr_blur[:b // 2, :, :, :]
                support_lr = lr_blur[b // 2:, :, :, :]
                query_hr = hr_blur[:b//2,:,:,:]
                support_hr = hr_blur[b // 2:, :, :, :]
                sr = self.model_meta(query_lr,query_hr, weights)

                loss = self.loss1(sr,query_label)
                loss_train[0] += loss.item()/self.meta_batch_size
                grads = torch.autograd.grad(loss, weights.values())
                fast_weights = OrderedDict(
                    (name, param - self.lr_task * grad) for ((name, param), grad) in zip(weights.items(), grads))


                for j in range(1,self.task_iter):
                    sr = self.model_meta(query_lr,query_hr, fast_weights)
                    loss = self.loss1(sr,query_label)
                    loss_train[j] += loss.item() / self.meta_batch_size
                    grads = torch.autograd.grad(loss, fast_weights.values())
                    fast_weights = OrderedDict(
                        (name, param - self.lr_task * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
                #***************support***********************
                sr = self.model_meta(support_lr,support_hr, fast_weights)
                loss = self.loss1(sr, support_label)
                grads = torch.autograd.grad(loss, weights.values())
                for ((name, _), g) in zip(meta_grads.items(), grads):
                    meta_grads[name] = meta_grads[name] + g/self.meta_batch_size


            hooks = []
            for (k, v) in self.model_meta.model.named_parameters():
                def get_closure():
                    key = k
                    def replace_grad(grad):
                        return meta_grads[key]
                    return replace_grad
                if v.requires_grad:
                    hooks.append(v.register_hook(get_closure()))

            sr = self.model_meta(query_lr,query_hr, weights)
            loss = self.loss(sr, query_label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Remove the hooks before next training phase
            for h in hooks:
                h.remove()

            timer.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                    'Loss [SR0 loss:{:.3f},SR1 loss:{:.3f},SR2 loss:{:.3f},SR3 loss:{:.3f},SR4 loss:{:.3f}]\t'
                    'Time [{:.1f}s]'.format(
                        epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                        loss_train[0],loss_train[1],loss_train[2],loss_train[3],loss_train[4],
                        timer.release(),
                    ))

        self.loss.end_log(len(self.loader_train))

        # save model
        if epoch > self.args.st_save_epoch or (epoch %10 ==0):
            target = self.model_meta.get_model()
            model_dict = target.state_dict()
            torch.save(
                model_dict,
                os.path.join(self.ckp.dir, 'model', 'model_meta_{}.pt'.format(epoch))
            )

            optimzer_dict = self.optimizer.state_dict()
            torch.save(
                optimzer_dict,
                os.path.join(self.ckp.dir, 'optimzer', 'optimzer_meta_{}.pt'.format(epoch))
            )

        target = self.model_meta.get_model()
        model_dict = target.state_dict()
        torch.save(
            model_dict,
            os.path.join(self.ckp.dir, 'model', 'model_meta_last.pt')
        )
        optimzer_dict = self.optimizer.state_dict()
        torch.save(
            optimzer_dict,
            os.path.join(self.ckp.dir, 'optimzer', 'optimzer_meta_last.pt')
        )


    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))

        timer_test = utility2.timer()
        self.model_meta.eval()
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
                    iter_psnr = [0]*(self.test_iter+1)
                    iter_ssim = [0]*(self.test_iter+1)
                    for idx_img, (hr, filename) in enumerate(d):
                        self.model_meta_copy.model.load_state_dict(self.model_meta.state_dict())
     
                        hr = hr.cuda()                      # b, c, h, w
                        hr = self.crop_border(hr, scale)
                        # inference
                        timer_test.tic()
                        hr_batch = hr
                        hr = utility2.quantize(hr, self.args.rgb_range)
                        lr_blur, hr_blur = degrade(hr_batch, random=False)

                        for iter in range(self.test_iter):
                            if iter == 0:
                                learning_rate = 1e-2
                            elif iter < 5:
                                learning_rate = 5e-3
                            else:
                                learning_rate = 1e-3

                            sr = self.model_meta_copy(lr_blur,hr_blur)

                            loss = self.loss1(sr, hr_batch)

                            sr = utility2.quantize(sr, self.args.rgb_range)

                            iter_psnr[iter] += utility2.calc_psnr(
                                sr, hr, scale, self.args.rgb_range,
                                benchmark=True
                            )
                            iter_ssim[iter] += utility2.calc_ssim(
                                (sr*255).round().clamp(0,255), (hr*255).round().clamp(0,255),scale,
                                benchmark=True
                            )

                            self.model_meta_copy.zero_grad()
                            loss.backward()

                            for param in self.model_meta_copy.parameters():
                                if param.requires_grad:
                                    param.data.sub_(param.grad.data * learning_rate)

                        sr = self.model_meta_copy(lr_blur,hr_blur)


                        timer_test.hold()

                        sr = utility2.quantize(sr, self.args.rgb_range)


                        # metrics
                        iter_psnr[-1] += utility2.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=True
                        )
                        iter_ssim[-1] += utility2.calc_ssim(
                            (sr*255).round().clamp(0,255), (hr*255).round().clamp(0,255),scale,
                            benchmark=True
                        )

                        # save results
                        if self.args.save_results:
                            save_list = [sr]
                            filename = filename[0]
                            self.ckp.save_results(filename, save_list, scale)


                    for t in range(self.test_iter+1):
                        print("iter:",t,",PSNR:",iter_psnr[t]/ len(self.loader_test),",SSIM:",iter_ssim[t]/ len(self.loader_test))

                    eval_psnr = iter_psnr[-1]
                    eval_ssim = iter_ssim[-1]

                    if len(self.test_res_psnr)>10:
                        self.test_res_psnr.pop(0)
                        self.test_res_ssim.pop(0)
                    self.test_res_psnr.append(eval_psnr / len(self.loader_test))
                    self.test_res_ssim.append(eval_ssim / len(self.loader_test))

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