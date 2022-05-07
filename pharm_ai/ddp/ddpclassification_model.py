import logging
import math
import os
import random
import warnings
from dataclasses import asdict

from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from simpletransformers.classification.classification_utils import (
    InputExample,
    LazyClassificationDataset,
    load_hf_dataset
)
from transformers.optimization import AdamW, Adafactor
from simpletransformers.classification import ClassificationModel

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False


logger = logging.getLogger(__name__)


class DDPClassificationModel(ClassificationModel):

    def __init__(
            self,
            model_type,
            model_name,
            tokenizer_type=None,
            tokenizer_name=None,
            num_labels=None,
            weight=None,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            onnx_execution_provider=None,
            **kwargs,
    ):
        """
        Initializes a DDP ClassificationModel model.
        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            tokenizer_type: The type of tokenizer (auto, bert, xlnet, xlm, roberta, distilbert, etc.) to use. If a string is passed, Simple Transformers will try to initialize a tokenizer class from the available MODEL_CLASSES.
                                Alternatively, a Tokenizer class (subclassed from PreTrainedTokenizer) can be passed.
            tokenizer_name: The name/path to the tokenizer. If the tokenizer_type is not specified, the model_type will be used to determine the type of the tokenizer.
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            onnx_execution_provider (optional): ExecutionProvider to use with ONNX Runtime. Will use CUDA (if use_cuda) or CPU (if use_cuda is False) by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        super().__init__(model_type, model_name, tokenizer_type, tokenizer_name,
                         num_labels, weight, args, use_cuda, cuda_device, onnx_execution_provider, **kwargs)
        self.args.use_multiprocessing = False
        self.args.use_multiprocessing_for_evaluation = False
        if self.args.n_gpu == 1:
            raise ValueError("You are using DDP with single GPU.")

    def train_model(
            self,
            train_df,
            multi_label=False,
            output_dir=None,
            show_running_loss=True,
            args=None,
            eval_df=None,
            verbose=True,
            **kwargs,
    ):
        """
        Trains the model using 'train_df'
        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"
        if args:
            self.args.update_from_dict(args)

        if self.args.evaluate_during_training and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if (
                os.path.exists(output_dir)
                and os.listdir(output_dir)
                and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(
                    output_dir
                )
            )

        if self.args.use_hf_datasets:
            if self.args.sliding_window:
                raise ValueError(
                    "HuggingFace Datasets cannot be used with sliding window."
                )
            if self.args.model_type == "layoutlm":
                raise NotImplementedError(
                    "HuggingFace Datasets support is not implemented for LayoutLM models"
                )
            train_dataset = load_hf_dataset(
                train_df, self.tokenizer, self.args, multi_label=multi_label
            )
        elif isinstance(train_df, str) and self.args.lazy_loading:
            if self.args.sliding_window:
                raise ValueError("Lazy loading cannot be used with sliding window.")
            if self.args.model_type == "layoutlm":
                raise NotImplementedError(
                    "Lazy loading is not implemented for LayoutLM models"
                )
            train_dataset = LazyClassificationDataset(
                train_df, self.tokenizer, self.args
            )
        else:
            if self.args.lazy_loading:
                raise ValueError(
                    "Input must be given as a path to a file when using lazy loading"
                )
            if "text" in train_df.columns and "labels" in train_df.columns:
                if self.args.model_type == "layoutlm":
                    train_examples = [
                        InputExample(i, text, None, label, x0, y0, x1, y1)
                        for i, (text, label, x0, y0, x1, y1) in enumerate(
                            zip(
                                train_df["text"].astype(str),
                                train_df["labels"],
                                train_df["x0"],
                                train_df["y0"],
                                train_df["x1"],
                                train_df["y1"],
                            )
                        )
                    ]
                else:
                    train_examples = (
                        train_df["text"].astype(str).tolist(),
                        train_df["labels"].tolist(),
                    )
            elif "text_a" in train_df.columns and "text_b" in train_df.columns:
                if self.args.model_type == "layoutlm":
                    raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                else:
                    train_examples = (
                        train_df["text_a"].astype(str).tolist(),
                        train_df["text_b"].astype(str).tolist(),
                        train_df["labels"].tolist(),
                    )
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                train_examples = (
                    train_df.iloc[:, 0].astype(str).tolist(),
                    train_df.iloc[:, 1].tolist(),
                )
            train_dataset = self.load_and_cache_examples(
                train_examples, verbose=verbose
            )

        os.makedirs(output_dir, exist_ok=True)

        os.environ['MASTER_ADDR'] = 'localhost'
        port = random.randint(10000, 20000)
        os.environ['MASTER_PORT'] = str(port)
        mp.spawn(self.train_each_proc, nprocs=self.args.n_gpu,
                 args=(train_dataset, multi_label, output_dir,
                       show_running_loss, eval_df, verbose, kwargs))

        self.save_model(model=self.model)

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_name, output_dir))

    def train_each_proc(self, process_index, train_dataset, *train_args):
        """
        A wrapper function of train() for each process of DDP.
        :param process_index: param train_dataset passed into train().
        :param train_dataset: The training set.
        :param train_args: other position arguments passed to train().
        :return: The same as train().
        """
        self._local_rank = process_index
        self._world_size = self.args.n_gpu
        self.train(train_dataset, *train_args[:-1], **train_args[-1])

    def train(
            self,
            train_dataset,
            multi_label=False,
            output_dir=None,
            show_running_loss=True,
            eval_data=None,
            verbose=True,
            **kwargs,
    ):
        if self.args.silent:
            show_running_loss = False
        self.device = torch.device(f"cuda:{self._local_rank}")
        self._move_model_to_device()
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self._world_size,
            rank=self._local_rank
        )
        num_labels = self.model.num_labels
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self._local_rank])
        setattr(self.model, "num_labels", num_labels)
        model = self.model
        args = self.args

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self._world_size,
            rank=self._local_rank
        )
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size // self._world_size,
            pin_memory=True,
            drop_last=False
        )

        # disable tensorboard when extra metrics are passed to avoid unintended behaviour in
        # multiprocessing setting
        disable_tb = len(kwargs) > 0
        if self._local_rank == 0 and not disable_tb:
            tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        else:
            tb_writer = None

        t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                               and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                               and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (warmup_steps if args.warmup_steps == 0 else args.warmup_steps)

        if 0 < args.save_after < 1:
            args.save_after = math.ceil(t_total * args.save_after)

        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
            print("Using Adafactor for T5")
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs),
            desc="Epoch",
            disable=args.silent or self._local_rank != 0,
            mininterval=0
        )
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        current_loss = "Initializing"
        stop_training = False

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                        len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                        len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(
                multi_label, **kwargs
            )

        if args.wandb_project and self._local_rank == 0:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for training.")
                wandb.init(
                    project=args.wandb_project,
                    config={**asdict(args), "repo": "simpletransformers"},
                    **args.wandb_kwargs,
                )
            wandb.watch(self.model)

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for epoch in train_iterator:
            model.train()
            train_sampler.set_epoch(epoch)
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            if self._local_rank == 0:
                train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs} on process {self._local_rank}",
                disable=args.silent or self._local_rank != 0,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs = self._get_inputs_dict(batch)
                if args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                loss_ = loss.clone()
                torch.distributed.barrier()
                torch.distributed.reduce(loss_, 0)
                current_loss = loss_.item() / self._world_size

                if show_running_loss and self._local_rank == 0:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0 and self._local_rank == 0:
                        # Log metrics
                        if not disable_tb:
                            tb_writer.add_scalar(
                                "lr", scheduler.get_last_lr()[0], global_step
                            )
                            tb_writer.add_scalar(
                                "loss",
                                (tr_loss - logging_loss) / args.logging_steps,
                                global_step,
                            )
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_last_lr()[0]
                                },
                                step=global_step
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0 and self._local_rank == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args.evaluate_during_training and (
                            args.evaluate_during_training_steps > 0
                            and global_step % args.evaluate_during_training_steps == 0
                    ):

                        results, _, _ = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent or self._local_rank != 0,
                            wandb_log=False,
                            **kwargs,
                        )

                        if self._local_rank == 0:
                            if not disable_tb:
                                for key, value in results.items():
                                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                            output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                            if args.save_eval_checkpoints:
                                self.save_model(
                                    output_dir_current,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )

                            stop_training, best_eval_metric, early_stopping_counter = self.logging_and_saving(
                                args,
                                results,
                                global_step,
                                train_iterator,
                                optimizer,
                                scheduler,
                                model,
                                training_progress_scores,
                                current_loss,
                                best_eval_metric,
                                verbose,
                                early_stopping_counter)

                        torch.distributed.barrier()
                        stop_training_tensor = torch.tensor([stop_training], device=self.device)
                        torch.distributed.broadcast(stop_training_tensor, src=0)
                        stop_training = bool(stop_training_tensor.cpu()[0])
                        if stop_training:
                            break

                        model.train()

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training and self._local_rank == 0:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch and self._local_rank == 0:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results, _, _ = self.eval_model(
                    eval_data,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent or self._local_rank != 0,
                    wandb_log=False,
                    **kwargs,
                )

                if self._local_rank == 0:
                    self.save_model(output_dir_current, optimizer, scheduler, results=results)

                    stop_training, best_eval_metric, early_stopping_counter = self.logging_and_saving(
                        args,
                        results,
                        global_step,
                        train_iterator,
                        optimizer,
                        scheduler,
                        model,
                        training_progress_scores,
                        current_loss,
                        best_eval_metric,
                        verbose,
                        early_stopping_counter)

                torch.distributed.barrier()
                stop_training_tensor = torch.tensor([stop_training], device=self.device)
                torch.distributed.broadcast(stop_training_tensor, src=0)
                stop_training = bool(stop_training_tensor.cpu()[0])
                if stop_training:
                    break

        # close tensorboard writer to avoid EOFError.
        if self._local_rank == 0:
            if not disable_tb:
                tb_writer.close()
            wandb.finish()

    def evaluate(
            self,
            eval_df,
            output_dir,
            multi_label=False,
            prefix="",
            verbose=True,
            silent=False,
            wandb_log=True,
            **kwargs,
    ):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}
        if self.args.use_hf_datasets:
            if self.args.sliding_window:
                raise ValueError(
                    "HuggingFace Datasets cannot be used with sliding window."
                )
            if self.args.model_type == "layoutlm":
                raise NotImplementedError(
                    "HuggingFace Datasets support is not implemented for LayoutLM models"
                )
            eval_dataset = load_hf_dataset(
                eval_df, self.tokenizer, self.args, multi_label=multi_label
            )
            eval_examples = None
        elif isinstance(eval_df, str) and self.args.lazy_loading:
            if self.args.model_type == "layoutlm":
                raise NotImplementedError(
                    "Lazy loading is not implemented for LayoutLM models"
                )
            eval_dataset = LazyClassificationDataset(eval_df, self.tokenizer, self.args)
            eval_examples = None
        else:
            if self.args.lazy_loading:
                raise ValueError(
                    "Input must be given as a path to a file when using lazy loading"
                )

            if "text" in eval_df.columns and "labels" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    eval_examples = [
                        InputExample(i, text, None, label, x0, y0, x1, y1)
                        for i, (text, label, x0, y0, x1, y1) in enumerate(
                            zip(
                                eval_df["text"].astype(str),
                                eval_df["labels"],
                                eval_df["x0"],
                                eval_df["y0"],
                                eval_df["x1"],
                                eval_df["y1"],
                            )
                        )
                    ]
                else:
                    eval_examples = (
                        eval_df["text"].astype(str).tolist(),
                        eval_df["labels"].tolist(),
                    )
            elif "text_a" in eval_df.columns and "text_b" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                else:
                    eval_examples = (
                        eval_df["text_a"].astype(str).tolist(),
                        eval_df["text_b"].astype(str).tolist(),
                        eval_df["labels"].tolist(),
                    )
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                eval_examples = (
                    eval_df.iloc[:, 0].astype(str).tolist(),
                    eval_df.iloc[:, 1].tolist(),
                )

            if args.sliding_window:
                eval_dataset, window_counts = self.load_and_cache_examples(
                    eval_examples, evaluate=True, verbose=verbose, silent=silent
                )
            else:
                eval_dataset = self.load_and_cache_examples(
                    eval_examples, evaluate=True, verbose=verbose, silent=silent
                )
        os.makedirs(eval_output_dir, exist_ok=True)

        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=self._world_size,
            rank=self._local_rank
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size // self._world_size,
            pin_memory=True,
            drop_last=False
        )

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(eval_dataloader)
        preds = np.empty((len(eval_dataset), self.num_labels))
        if multi_label:
            out_label_ids = np.empty((len(eval_dataset), self.num_labels))
        else:
            out_label_ids = np.empty((len(eval_dataset)))
        model.eval()

        if self.args.fp16:
            from torch.cuda import amp

        for i, batch in enumerate(
                tqdm(
                    eval_dataloader,
                    disable=args.silent or silent,
                    desc="Running Evaluation",
                )
        ):

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs)
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()

                torch.distributed.barrier()
                torch.distributed.reduce(tmp_eval_loss, 0)
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            gather_labels = self._gather_data(inputs["labels"])
            gather_logits = self._gather_data(logits)

            start_index = self.args.eval_batch_size * i
            end_index = (
                start_index + self.args.eval_batch_size
                if i != (n_batches - 1)
                else len(eval_dataset)
            )
            preds[start_index:end_index] = gather_logits.detach().cpu().numpy()[:end_index-start_index]
            out_label_ids[start_index:end_index] = (
                gather_labels.detach().cpu().numpy()[:end_index-start_index]
            )

        eval_loss = eval_loss / nb_eval_steps / self._world_size

        model_outputs = None
        wrong = None
        if self._local_rank == 0:
            if args.sliding_window:
                count = 0
                window_ranges = []
                for n_windows in window_counts:
                    window_ranges.append([count, count + n_windows])
                    count += n_windows

                preds = [
                    preds[window_range[0]: window_range[1]]
                    for window_range in window_ranges
                ]
                out_label_ids = [
                    out_label_ids[i]
                    for i in range(len(out_label_ids))
                    if i in [window[0] for window in window_ranges]
                ]

                model_outputs = preds

                preds = [np.argmax(pred, axis=1) for pred in preds]
                final_preds = []
                for pred_row in preds:
                    val_freqs_desc = Counter(pred_row).most_common()
                    if (
                            len(val_freqs_desc) > 1
                            and val_freqs_desc[0][1] == val_freqs_desc[1][1]
                    ):
                        final_preds.append(args.tie_value)
                    else:
                        final_preds.append(val_freqs_desc[0][0])
                preds = np.array(final_preds)
            elif not multi_label and args.regression is True:
                preds = np.squeeze(preds)
                model_outputs = preds
            else:
                model_outputs = preds

                if not multi_label:
                    preds = np.argmax(preds, axis=1)

            result, wrong = self.compute_metrics(
                preds, model_outputs, out_label_ids, eval_examples, **kwargs
            )
            result["eval_loss"] = eval_loss
            results.update(result)

            output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(result.keys()):
                    writer.write("{} = {}\n".format(key, str(result[key])))

            if (
                    self.args.wandb_project
                    and wandb_log
                    and not multi_label
                    and not self.args.regression
            ):
                if not wandb.setup().settings.sweep_id:
                    logger.info(" Initializing WandB run for evaluation.")
                    wandb.init(
                        project=args.wandb_project,
                        config={**asdict(args), "repo": "simpletransformers"},
                        **args.wandb_kwargs,
                    )
                if not args.labels_map:
                    self.args.labels_map = {i: i for i in range(self.num_labels)}

                labels_list = sorted(list(self.args.labels_map.keys()))
                inverse_labels_map = {
                    value: key for key, value in self.args.labels_map.items()
                }

                truth = [inverse_labels_map[out] for out in out_label_ids]

                # Confusion Matrix
                wandb.sklearn.plot_confusion_matrix(
                    truth, [inverse_labels_map[pred] for pred in preds], labels=labels_list,
                )

                if not self.args.sliding_window:
                    # ROC`
                    wandb.log({"roc": wandb.plots.ROC(truth, model_outputs, labels_list)})

                    # Precision Recall
                    wandb.log(
                        {
                            "pr": wandb.plots.precision_recall(
                                truth, model_outputs, labels_list
                            )
                        }
                    )

        return results, model_outputs, wrong

    def _gather_data(self, data):
        """
        Gather data across GPUs
        Args:
            data: data on each GPU

        Returns:
            gathered data
        """
        gather_data = [torch.ones_like(data) for _ in range(self._world_size)]
        torch.distributed.all_gather(gather_data, data)
        return torch.cat(gather_data)

    def logging_and_saving(
            self,
            args,
            results,
            global_step,
            train_iterator,
            optimizer,
            scheduler,
            model,
            training_progress_scores,
            current_loss,
            best_eval_metric,
            verbose,
            early_stopping_counter):
        training_progress_scores["global_step"].append(global_step)
        training_progress_scores["train_loss"].append(current_loss)
        for key in results:
            training_progress_scores[key].append(results[key])
        report = pd.DataFrame(training_progress_scores)
        report.to_csv(
            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
        )

        if args.wandb_project or self.is_sweeping:
            wandb.log(self._get_last_metrics(training_progress_scores), step=global_step)

        stop_training = False
        if global_step > args.save_after:
            if not best_eval_metric:
                best_eval_metric = results[args.early_stopping_metric]
                self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)

            if args.early_stopping_metric_minimize:
                if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                    early_stopping_counter = 0
                else:
                    stop_training, early_stopping_counter = \
                        self.check_early_stopping(early_stopping_counter, args, train_iterator, verbose)
            else:
                if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                    early_stopping_counter = 0
                else:
                    stop_training, early_stopping_counter = \
                        self.check_early_stopping(early_stopping_counter, args, train_iterator, verbose)

        return stop_training, best_eval_metric, early_stopping_counter

    def check_early_stopping(self, early_stopping_counter, args, train_iterator, verbose):
        stop_training = False
        if args.use_early_stopping:
            if early_stopping_counter < args.early_stopping_patience:
                early_stopping_counter += 1
                if verbose:
                    logger.info(f" No improvement in {args.early_stopping_metric}")
                    logger.info(f" Current step: {early_stopping_counter}")
                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
            else:
                if verbose:
                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                    logger.info(" Training terminated.")
                    train_iterator.close()
                stop_training = True
        return stop_training, early_stopping_counter