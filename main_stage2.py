from option import args
import torch
import utility
import data
import model_meta
import model
import loss
from trainer_stage2 import Trainer


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        model_meta = model_meta.Model(args, checkpoint)
        model_meta_copy = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model_meta, model_meta_copy, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()


        checkpoint.done()