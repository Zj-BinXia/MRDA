import os
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util
import numpy as np


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.test_res_psnr = []
        self.test_res_ssim = []
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

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
        self.model.train()

        degrade = util.BicubicPreprocessing(
            self.scale[0],
            rgb_range = self.args.rgb_range
        )

        timer = utility.timer()
        losses_sr = utility.AverageMeter()

        for batch, (hr, _,) in enumerate(self.loader_train):
            hr = hr.cuda()                              # b, c, h, w
            lr,lr_bic = degrade(hr)                 # b, c, h, w
            #b_kernels, noise_level = degradation
            self.optimizer.zero_grad()

            timer.tic()
            # forward
            ## train the whole network
            sr = self.model(lr,lr_bic)
            loss_SR = self.loss(sr, hr)
            loss = loss_SR
            losses_sr.update(loss_SR.item())

            # backward
            loss.backward()
            self.optimizer.step()
            timer.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                    'Loss [SR loss:{:.3f}]\t'
                    'Time [{:.1f}s]'.format(
                        epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                        losses_sr.avg,
                        timer.release(),
                    ))

        self.loss.end_log(len(self.loader_train))

        # save model
        if epoch > self.args.st_save_epoch or epoch%30==0:
            target = self.model.get_model()
            model_dict = target.state_dict()
            torch.save(
                model_dict,
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
            )

        target = self.model.get_model()
        model_dict = target.state_dict()
        torch.save(
            model_dict,
            os.path.join(self.ckp.dir, 'model', 'model_last.pt')
        )

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        degrade = util.BicubicPreprocessing(
                    self.scale[0],
                    rgb_range=self.args.rgb_range
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
                        lr,lr_bic = degrade(hr)    # b, c, h, w

                        # inference
                        timer_test.tic()
                        sr = self.model(lr,lr_bic)
                        timer_test.hold()

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

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >=  self.args.epochs_sr

