import logging
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

from data.recognizer_datasets.dataset_mapper import DatasetMapperWithBasis
from fots_recognizer.checkpoint import AdetCheckpointer
from fots_recognizer.config import get_cfg

class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader` method.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            # model = DistributedDataParallel(
            #     model, find_unused_parameters = True, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            # )
        super(DefaultTrainer, self).__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = AdetCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("fots_recognizer.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="fots_recognizer")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
