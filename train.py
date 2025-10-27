import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers.optimization import get_linear_schedule_with_warmup

from src.model import JobMatchingMultiTaskModel
from src.args import get_args



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    acc = torch.accelerator.current_accelerator()
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def reduce_mean_tensor(tensor, world_size):
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / world_size
    return tensor


def run_spawn(train_fn, world_size, model_args, data_args):
    mp.spawn(train_fn, args=(world_size, model_args, data_args), nprocs=world_size, join=True)


def download_dataset(dataset_name: str, tokenizer: PreTrainedTokenizerBase):
    df = pd.read_csv(dataset_name)
    dataset = load_dataset("csv", data_files=dataset_name)

    train_dataset = dataset["train"].map(
        prepare_features, 
        batched=True,
        fn_kwargs={"column_names": dataset["train"].column_names, "tokenizer": tokenizer}, 
        remove_columns=dataset["train"].column_names
    )

    return train_dataset


def train(rank, world_size, model_args, data_args):
    print(f"[rank {rank}] starting")
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    train_dataset = download_dataset(data_args.dataset_name, tokenizer)

    config_kwargs = {"cache_dir": model_args.cache_dir, "revision": model_args.model_revision, "use_auth_token": True if model_args.use_auth_token else None}
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)  

    sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        num_workers=data_args.preprocessing_num_workers,
        sampler=sampler,
        collate_fn=default_data_collator,
        drop_last=True
    )    
    
    model = JobMatchingMultiTaskModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        model_args=model_args,
        data_args=data_args
    )
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    if torch.cuda.is_available():
        ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    else:
        ddp_model = DDP(model)
    ddp_model.train()

    optimizer = torch.optim.AdamW(
        params=ddp_model.parameters(), 
        lr=model_args.lr
    )        
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=model_args.num_warmup_steps,
        num_training_steps=len(train_dataloader) * model_args.epochs
    )
    
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(model_args.epochs):        
        sampler.set_epoch(epoch)
        epoch_losses = []

        if rank == 0:
            progress = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        else:
            progress = train_dataloader 

        changes = []
        for batch_idx, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            jd_labels = batch["jd_labels"].to(device, non_blocking=True)
            jf_labels = batch["jf_labels"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                total_loss_metadata = ddp_model(input_ids, attention_mask, jd_labels=jd_labels, jf_labels=jf_labels)           
                loss = total_loss_metadata['total_loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=10.0)

            if rank == 0 and batch_idx % 100 == 0:
                total_norm = 0.0
                for p in ddp_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"[rank {rank}] grad_norm={total_norm:.6f}")
            
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.detach().cpu())
            
            if rank == 0 and len(epoch_losses) % 10 == 0:
                print(f"Loss: {loss}")
        
        # === Average losses across GPUs ===
        if len(epoch_losses) > 0:
            avg_loss_local = torch.stack(epoch_losses).mean().to(device)
        else:
            avg_loss_local = torch.tensor(0.0, device=device)

        avg_loss_global = reduce_mean_tensor(avg_loss_local, world_size)

        if rank == 0:
            print(f"Epoch {epoch} | global avg loss = {avg_loss_global.item():.6f}")

        lr_scheduler.step()
    cleanup()
    print(f"[rank {rank}] finished")


if __name__ == "__main__":
    model_args, data_args = get_args()
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    run_spawn(train, world_size, model_args, data_args)