"""Learning-rate schedules built from the configuration."""

import torch.optim.lr_scheduler as lr_scheduler

def create_learning_rate_fn(optimizer, max_steps, args, debug=False):
    """Create learning rate schedule."""
    if args.type == "none":
        return None

    if args.warmup > 0:
        warmup_start_factor = 1e-16
    else:
        warmup_start_factor = 1.0

    warmup_fn = lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=args.warmup,
        verbose=debug,
    )

    if args.type == "cosine":
        cosine_steps = max(max_steps - args.warmup, 1)
        decay_fn = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, verbose=debug
        )
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    elif args.type == "cosine-hlfperiod":
        cosine_steps = max(max_steps - args.warmup, 1) * 2
        decay_fn = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, verbose=debug
        )
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    else:
        raise NotImplementedError

    schedule_fn = lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=milestones, verbose=debug
    )

    return schedule_fn
