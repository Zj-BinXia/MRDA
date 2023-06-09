import os
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util2
import numpy as np
import torch.nn as nn


class Trainer():
    def __init__(self, args, loader, model_ST, model_TA,model_meta_copy, my_loss, ckp):
        self.is_first =True
        self.args = args
        self.scale = args.scale
        self.test_res_psnr = []
        self.test_res_ssim = []
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model_ST = model_ST
        self.model_TA = model_TA
        self.model_meta_copy = model_meta_copy
        for k,v in self.model_meta_copy.named_parameters():
            if "tail" in k:
                v.requires_grad=False
        self.model_meta_copy_state = self.model_meta_copy.state_dict()
        self.model_Est = torch.nn.DataParallel(self.model_ST.get_model().E_st, range(self.args.n_GPUs))
        self.model_Eta = torch.nn.DataParallel(self.model_TA.get_model().E, range(self.args.n_GPUs))
        self.loss1 = nn.L1Loss()
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model_ST)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.task_iter = args.task_iter
        self.test_iter = args.test_iter
        self.lr_task = args.lr_task
        self.temperature = args.temperature

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model_ST.train()

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

        timer = utility.timer()
        losses_sr, losses_distill_distribution, losses_distill_abs = utility.AverageMeter(),utility.AverageMeter(),utility.AverageMeter()


        for batch, (hr, _) in enumerate(self.loader_train):
            hr = hr.cuda()  # b, c, h, w
            # b,c,h,w = hr.shape
            timer.tic()
            # loss_all = 0
            lr_blur, hr_blur = degrade(hr)  # b, c, h, w
            self.model_meta_copy.get_model().load_state_dict(self.model_meta_copy_state)

            for iter in range(self.test_iter):
                if iter == 0:
                    learning_rate = 1e-2
                elif iter < 5:
                    learning_rate = 5e-3
                else:
                    learning_rate = 1e-3
                sr, _ = self.model_meta_copy(lr_blur, hr_blur)
                loss = self.loss1(sr, hr)
                self.model_meta_copy.zero_grad()
                loss.backward()

                for param in self.model_meta_copy.parameters():
                    if param.requires_grad:
                        param.data.sub_(param.grad.data * learning_rate)

            _, deg_repre = self.model_meta_copy(lr_blur, hr_blur)
            _, T_fea = self.model_Eta(deg_repre)

            loss_distill_dis = 0
            loss_distill_abs = 0

            if epoch <= self.args.epochs_encoder:
                if self.is_first:
                    _,  S_fea = self.model_ST(lr_blur)
                    self.is_first = False
                else:
                    _, S_fea = self.model_Est(lr_blur)
                for i in range(len(T_fea)):
                    student_distance = F.log_softmax(S_fea[i] / self.temperature, dim=1)
                    teacher_distance = F.softmax(T_fea[i]/ self.temperature, dim=1)
                    loss_distill_dis += F.kl_div(
                        student_distance, teacher_distance, reduction='batchmean')
                    loss_distill_abs += nn.L1Loss()(S_fea[i], T_fea[i])
                losses_distill_distribution.update(loss_distill_dis.item())
                losses_distill_abs.update(loss_distill_abs.item())
                loss = loss_distill_dis + 0.1*loss_distill_abs
            else:
                sr, S_fea = self.model_ST(lr_blur)
                loss_SR = self.loss(sr, hr)
                for i in range(len(T_fea)):
                    student_distance = F.log_softmax(S_fea[i] / self.temperature, dim=1)
                    teacher_distance = F.softmax(T_fea[i] / self.temperature, dim=1)
                    loss_distill_dis += F.kl_div(
                        student_distance, teacher_distance, reduction='batchmean')
                    loss_distill_abs += nn.L1Loss()(S_fea[i], T_fea[i])
                losses_distill_distribution.update(loss_distill_dis.item())
                losses_distill_abs.update(loss_distill_abs.item())
                loss = loss_SR + loss_distill_dis + 0.1 * loss_distill_abs
                losses_sr.update(loss_SR.item())

                # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            timer.hold()

            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [ distill_dis loss:{:.3f}, distill_abs loss:{:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_distill_distribution.avg, losses_distill_abs.avg,
                            timer.release(),
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [SR loss:{:.3f}, distill_dis loss:{:.3f}, distill_abs loss:{:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_sr.avg, losses_distill_distribution.avg, losses_distill_abs.avg,
                            timer.release(),
                        ))

        self.loss.end_log(len(self.loader_train))

        # save model
        if epoch > self.args.st_save_epoch or epoch%30==0:
            target = self.model_ST.get_model()
            model_dict = target.state_dict()
            torch.save(
                model_dict,
                os.path.join(self.ckp.dir, 'model', 'model_ST_{}.pt'.format(epoch))
            )

        target = self.model_ST.get_model()
        model_dict = target.state_dict()
        torch.save(
            model_dict,
            os.path.join(self.ckp.dir, 'model', 'model_ST_last.pt')
        )

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model_ST.eval()
        t = 0
        timer_test = utility.timer()

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

        with torch.no_grad():
            for idx_scale, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    eval_psnr = 0
                    eval_ssim = 0
                    for idx_img, (hr, filename) in enumerate(d):
                        hr = hr.cuda()                      # b, c, h, w
                        hr = self.crop_border(hr, scale)
                        lr_blur, hr_blur = degrade(hr, random=False)    # b, c, h, w

                        # inference
                        torch.cuda.synchronize()
                        timer_test.tic()
                        sr = self.model_ST(lr_blur)
                        torch.cuda.synchronize()
                        timer_test.hold()
                        t0 = timer_test.release()
                        print("idx:", idx_img, ",time consuming:", t0)
                        t += t0

                        sr = utility.quantize(sr, self.args.rgb_range)
                        hr = utility.quantize(hr, self.args.rgb_range)

                        # metrics
                        eval_psnr += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=True
                        )
                        eval_ssim += utility.calc_ssim(
                            (sr * 255).round().clamp(0, 255), (hr * 255).round().clamp(0, 255), scale,
                            benchmark=True
                        )

                        # save results
                        if self.args.save_results:
                            save_list = [sr]
                            filename = filename[0]
                            self.ckp.save_results(filename, save_list, scale)

                if len(self.test_res_psnr) > 10:
                    self.test_res_psnr.pop(0)
                    self.test_res_ssim.pop(0)
                self.test_res_psnr.append(eval_psnr / len(d))
                self.test_res_ssim.append(eval_ssim / len(d))
                print("All time consuming:", t / len(d))

                self.ckp.log[-1, idx_scale] = eval_psnr / len(d)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} mean_PSNR: {:.3f} mean_SSIM: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(d),
                        eval_ssim / len(d),
                        np.mean(self.test_res_psnr),
                        np.mean(self.test_res_ssim)
                    ))

    def crop_border(self, img_hr, scale):
        b, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >=  self.args.epochs_sr

