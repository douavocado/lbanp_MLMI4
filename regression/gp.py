# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2022, Tung Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TNP (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen 
####################################################################################

import os
import os.path as osp
import argparse
import random
import yaml

import torch
import time
from attrdict import AttrDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.ground_truth import GaussianProcess

from data.gp import *
from data.sawtooth import SawtoothSampler
from data.mixture import MixtureSampler
from utils.misc import load_module
from utils.paths import results_path, evalsets_path, testsets_path
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument('--device', type=str, default="cuda")
    # Experiment
    parser.add_argument('--mode', default='train')
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_from', type=str, default=None)

    # Data
    parser.add_argument('--max_num_points', type=int, default=50)

    # Model
    parser.add_argument('--model', type=str, default="tnpd")

    # Train
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--task', type=str, default='gp')

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_batches', type=int, default=3000)
    parser.add_argument('--eval_num_bs', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)


    # LBANP Arguments
    parser.add_argument('--num_latents', type=int, default=8)
    parser.add_argument('--num_latents_per_layer', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--emb_depth', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dropout', type=int, default=0.0)
    parser.add_argument('--num_layers', type=int, default=6)

    # OOD settings
    parser.add_argument('--eval_kernel', type=str, default='rbf')
    parser.add_argument('--t_noise', type=float, default=None)

    # AR settings
    parser.add_argument('--use_ar', action='store_true')
    parser.add_argument('--small_only', action='store_true')
    parser.add_argument('--track_diff', action='store_true')
    parser.add_argument('--ground_truth', action='store_true')

    args = parser.parse_args()

    if args.expid is not None:
        args.root = osp.join(results_path, 'gp', args.model, args.expid)
    else:
        args.root = osp.join(results_path, 'gp', args.model)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/gp/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for key, val in vars(args).items(): # Override the default arguments
        if key in config:
            config[key] = val
            print(f"Overriding argument {key}: {config[key]}")

    if args.pretrain:
        assert args.model == 'tnpa'
        config['pretrain'] = args.pretrain

    model = model_cls(**config)
    device = torch.device(args.device)
    model.to(device)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'test':
        eval_testset(args, model)

def visualize_prediction(model, sampler, args, step, device):
    """
    Visualize model predictions on a random sample from the sampler.
    
    Args:
        model: The trained model
        sampler: Data sampler
        args: Command line arguments
        step: Current training step
        device: Computation device
    """
    model.eval()
    with torch.no_grad():
        # Generate a single batch sample for visualization
        batch = sampler.sample(batch_size=1, max_num_points=args.max_num_points, device=device)
        
        # Get model predictions
        # if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
        #     outs = model(batch, num_samples=50)  # Use more samples for better uncertainty estimation
        # else:
        outs = model.predict(batch.xc, batch.yc, batch.xt)
        
        # Extract data for plotting
        x_context = batch['xc'][0,:,0].cpu().numpy()
        y_context = batch['yc'][0,:,0].cpu().numpy()
        x_target = batch['xt'][0,:,0].cpu().numpy()
        y_target = batch['yt'][0,:,0].cpu().numpy()
        
        # Get predictions
        mean = outs.loc[0, :,0].cpu().numpy()
        std = outs.scale[0, :,0].cpu().numpy()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot context points
        plt.scatter(x_context, y_context, color='blue', marker='o', s=50, label='Context')
        
        # Plot target points
        plt.scatter(x_target, y_target, color='green', marker='x', s=30, label='Target (True)')
        
        # Sort x_target for smooth curve plotting
        sort_idx = np.argsort(x_target)
        x_target_sorted = x_target[sort_idx]
        mean_sorted = mean[sort_idx]
        std_sorted = std[sort_idx]
        
        # Plot mean prediction
        plt.plot(x_target_sorted, mean_sorted, color='red', label='Prediction Mean')
        
        # Plot confidence intervals (2 standard deviations)
        plt.fill_between(
            x_target_sorted, 
            mean_sorted - 2 * std_sorted, 
            mean_sorted + 2 * std_sorted, 
            color='red', alpha=0.2, label='95% Confidence'
        )
        
        plt.title(f'Model Predictions at Step {step}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(args.root, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(plots_dir, f'prediction_step_{step}.png'), dpi=150)
        plt.close()
    
    model.train()

def train(args, model):
    device = torch.device(args.device)
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args, device)

    torch.manual_seed(args.train_seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.train_seed)

    if args.task == 'gp':
        sampler = GPSampler(RBFKernel())
    elif args.task == 'sawtooth':
        sampler = SawtoothSampler()
    elif args.task == 'mixture':
        sampler = MixtureSampler()
    else:
        raise ValueError(f'Invalid task {args.task}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps)

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    elif args.resume_from:
        ckpt = torch.load(args.resume_from)
        model.load_state_dict(ckpt.model)
        # optimizer.load_state_dict(ckpt.optimizer)
        # scheduler.load_state_dict(ckpt.scheduler)
        logfilename = os.path.join(args.root, f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1
    else:
        logfilename = os.path.join(args.root,
                f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info(f"Experiment: {args.model}-{args.expid}")
        logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    for step in range(start_step, args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        # frequency range for sawtooth gradually increases from initial 0.5 to 2 to 2 to 4 by the end of training
        # same for number of context points. Starts high at random int between 30 and 100
        # and decreases to between 0 and 30 by the end of training
        # number of target points starts at 30 at start of training and increases to 100 by the end of training
        if args.task == 'sawtooth':
            # freq_range_high = step / args.num_steps * (4 - 2) + 2
            # freq_range_low = step / args.num_steps * (2 - 0.5) + 0.5
            freq_range_high = 4
            freq_range_low = 2
            # num_ctx_high = int(100-step / args.num_steps * (100 - 30))
            # num_ctx_low = int(30-step / args.num_steps * (30 - 0))
            # num_tar = int(30+step / args.num_steps * (100 - 30))
            num_ctx_high = 30
            num_ctx_low = 0
            num_tar = 100

            batch = sampler.sample(
                batch_size=args.train_batch_size,
                max_num_points=args.max_num_points,
                freq_range=(freq_range_low, freq_range_high),
                num_ctx=random.randint(num_ctx_low, num_ctx_high),
                num_tar=num_tar,
                device=device)
        else:
            batch = sampler.sample(
                batch_size=args.train_batch_size,
                max_num_points=args.max_num_points,
                device=device)
        
        if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
            outs = model(batch, num_samples=args.train_num_samples)
        else:
            outs = model(batch)

        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)


        if step % args.print_freq == 0:
            line = f'{args.model}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info() + "\n"
            if args.task == 'sawtooth':
                line += f"num_ctx_high: {num_ctx_high}, num_ctx_low: {num_ctx_low}, num_tar: {num_tar}\n"
                line += f"freq_range_high: {freq_range_high}, freq_range_low: {freq_range_low}\n"
            logger.info(line)

            if step % args.eval_freq == 0:
                line = eval(args, model)
                logger.info(line + '\n')

            ravg.reset()
        
        # Plot predictions visualization every 1000 steps
        if step % 1000 == 0:
            visualize_prediction(model, sampler, args, step, device)

        if step % args.save_freq == 0 or step == args.num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, os.path.join(args.root, 'ckpt.tar'))
    args.mode = 'eval'
    eval(args, model)

def get_eval_path(args):
    path = osp.join(evalsets_path, args.task)
    filename = f'{args.eval_kernel}-seed{args.eval_seed}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    if args.ground_truth:
        filename += '_gt'
    filename += '.tar'
    return path, filename

def get_test_path(args):
    path = osp.join(testsets_path, args.task)
    filename = f'{args.eval_kernel}-seed{args.eval_seed}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    return path, filename

def gen_evalset(args, device):
    if args.task == 'gp':
        if args.eval_kernel == 'rbf':
            kernel = RBFKernel()
        elif args.eval_kernel == 'matern':
            kernel = Matern52Kernel()
        elif args.eval_kernel == 'periodic':
            kernel = PeriodicKernel()
        else:
            raise ValueError(f'Invalid kernel {args.eval_kernel}')
        print(f"Generating Evaluation Sets with {args.eval_kernel} kernel")

        sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)
    elif args.task == 'sawtooth':
        sampler = SawtoothSampler()
        print(f"Generating Evaluation Sets with Sawtooth sampler")
    elif args.task == 'mixture':
        sampler = MixtureSampler()
        print(f"Generating Evaluation Sets with Mixture sampler")
    else:
        raise ValueError(f'Invalid task {args.task}')
    
    batches = []
    for i in tqdm(range(args.eval_num_batches), ascii=True):
        if args.ground_truth:
            batches.append(sampler.sample(
                batch_size=args.eval_batch_size,
                max_num_points=args.max_num_points,
                device=device, return_params=True))
        else:
            batches.append(sampler.sample(
                batch_size=args.eval_batch_size,
                max_num_points=args.max_num_points,
                device=device))

    torch.manual_seed(time.time())
    if args.device == "cuda":   
        torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(args)
    
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))

def gen_testset(args, device):
    if args.task == 'gp':
        if args.eval_kernel == 'rbf':
            kernel = RBFKernel()
        elif args.eval_kernel == 'matern':
            kernel = Matern52Kernel()
        elif args.eval_kernel == 'periodic':
            kernel = PeriodicKernel()
        else:
            raise ValueError(f'Invalid kernel {args.eval_kernel}')
        print(f"Generating Evaluation Sets with {args.eval_kernel} kernel")

        sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)
    elif args.task == 'sawtooth':
        sampler = SawtoothSampler()
        print(f"Generating Evaluation Sets with Sawtooth sampler")
    elif args.task == 'mixture':
        sampler = MixtureSampler()
        print(f"Generating Evaluation Sets with Mixture sampler")
    else:
        raise ValueError(f'Invalid task {args.task}')

    batches = []
    for i in tqdm(range(args.eval_num_batches), ascii=True):
        batches.append(sampler.sample(
            batch_size=args.eval_batch_size,
            max_num_points=args.max_num_points,
            xt_range=(2, 6),
            device=device))

    torch.manual_seed(time.time())
    if args.device == "cuda":   
        torch.cuda.manual_seed(time.time())

    path, filename = get_test_path(args)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))

def eval(args, model):
    device = torch.device(args.device)
    # eval a trained model on log-likelihood
    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location=args.device)
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f'eval_{args.eval_kernel}'
            if args.t_noise is not None:
                eval_logfile += f'_tn_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args, device)
    
    print(f"Loading evaluation batches from {path}/{filename}")
    eval_batches = torch.load(osp.join(path, filename))

    if args.mode == "eval":
        torch.manual_seed(args.eval_seed)
        if args.device == "cuda":
            torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    diff_tracker = {}
    diff_tracker_count = {}
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            if args.small_only:
                if batch.xt.shape[1] >= 6:
                    continue # only give small samples
            for key, val in batch.items():
                batch[key] = val.to(device)
            if args.ground_truth:
                ground_truth_model = GaussianProcess(dim_x=1, dim_y=1)
                outs = ground_truth_model(batch)
            elif args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, args.eval_num_samples)
            else:
                if args.use_ar:
                    if args.track_diff:
                        outs, out_dic = model(batch, use_ar=args.use_ar, track_diff=True)
                        if out_dic["context_size"] not in diff_tracker:
                            diff_tracker[out_dic["context_size"]] = out_dic["diff"]
                        else:
                            diff_tracker[out_dic["context_size"]] += out_dic["diff"]
                        if out_dic["context_size"] not in diff_tracker_count:
                            diff_tracker_count[out_dic["context_size"]] = batch["xc"].shape[1]
                        else:
                            diff_tracker_count[out_dic["context_size"]] += batch["xc"].shape[1]
                    else:
                        outs = model(batch, use_ar=args.use_ar, track_diff=False)

                else:
                    outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    if args.device == "cuda":   
        torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate average differences
    average_diff = {cat: diff_tracker[cat] / diff_tracker_count[cat] for cat in diff_tracker}
    
    # Sort the data by context size
    context_sizes = sorted(average_diff.keys())
    diffs = [average_diff[size] for size in context_sizes]
    counts = [diff_tracker_count[size] for size in context_sizes]
    
    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot average differences
    ax1.plot(context_sizes, diffs, 'b-', marker='o', linewidth=2, label='Average Difference')
    ax1.set_xlabel('Context Size')
    ax1.set_ylabel('Average Difference', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for counts
    ax2 = ax1.twinx()
    
    # Plot counts as area plot
    ax2.plot(context_sizes, counts, 'r-', alpha=0.5, linewidth=1)
    ax2.fill_between(context_sizes, counts, alpha=0.2, color='r', label='Sample Count')
    ax2.set_ylabel('Sample Count', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add title and legend
    plt.title('Average Difference vs Context Size')
    fig.tight_layout()
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Save the plot
    plt.savefig('context_size_diff_plot.png', dpi=300, bbox_inches='tight')
    # Create a text file with LaTeX table code
    with open('context_size_table.tex', 'w') as f:
        # Write LaTeX table header
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Context Size & Average Difference & Sample Count \\\\\n")
        f.write("\\midrule\n")
        
        # Group context sizes in intervals of five
        intervals = {}
        for size in context_sizes:
            interval_key = f"{(size-1)//5*5+1}-{(size-1)//5*5+5}"
            if interval_key not in intervals:
                intervals[interval_key] = {"diff_sum": 0, "count_sum": 0, "num_sizes": 0}
            
            intervals[interval_key]["diff_sum"] += average_diff[size]
            intervals[interval_key]["count_sum"] += diff_tracker_count[size]
            intervals[interval_key]["num_sizes"] += 1
        
        # Write each interval to the table
        for interval, data in sorted(intervals.items(), key=lambda x: int(x[0].split('-')[0])):
            avg_diff = data["diff_sum"] / data["num_sizes"]
            avg_count = data["count_sum"] / data["num_sizes"]
            f.write(f"{interval} & {avg_diff:.4f} & {int(avg_count)} \\\\\n")
        
        # Write LaTeX table footer
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Average difference between autoregressive and marginal log-likelihood by context size intervals}\n")
        f.write("\\label{tab:context_size_diff}\n")
        f.write("\\end{table}\n")
    
    # Also create a more detailed plot showing the grouped data
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare grouped data for plotting
    interval_labels = []
    interval_diffs = []
    
    for interval, data in sorted(intervals.items(), key=lambda x: int(x[0].split('-')[0])):
        interval_labels.append(interval)
        interval_diffs.append(data["diff_sum"] / data["num_sizes"])
    
    # Plot the grouped data
    ax.bar(interval_labels, interval_diffs, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Context Size Intervals')
    ax.set_ylabel('Average Difference')
    ax.set_title('Average Difference by Context Size Intervals')
    
    # Add values on top of bars
    for i, v in enumerate(interval_diffs):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('context_size_intervals_plot.png', dpi=300, bbox_inches='tight')
    print(average_diff)
    print(diff_tracker_count)
    return line


def eval_testset(args, model):
    device = torch.device(args.device)
    # eval a trained model on log-likelihood
    if args.mode == 'test':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location=args.device)
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f'test_{args.eval_kernel}'
            if args.t_noise is not None:
                eval_logfile += f'_tn_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path, filename = get_test_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_testset(args, device)
    test_batches = torch.load(osp.join(path, filename))

    if args.mode == "eval":
        torch.manual_seed(args.eval_seed)
        if args.device == "cuda":
            torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.to(device)
            if args.ground_truth:
                ground_truth_model = GaussianProcess(dim_x=1, dim_y=1)
                outs = ground_truth_model(batch)
            elif args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, args.eval_num_samples)
            else:
                outs = model(batch, use_ar=args.use_ar)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    if args.device == "cuda":   
        torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line

if __name__ == '__main__':
    main()
