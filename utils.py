import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import MaxNLocator

import seaborn as sns
import pandas as pd


# define how your plots look:
def setup_axes(fig, rect, theta, radius):

    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degree
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

    # Find grid values appropriate for the coordinate (degree).
    # The argument is an approximate number of grids.
    grid_locator1 = angle_helper.LocatorD(10)

    # And also use an appropriate formatter:
    tick_formatter1 = angle_helper.FormatterDMS()

    # set up number of ticks for the r-axis
    grid_locator2 = MaxNLocator(5)

    # the extremes are passed to the function
    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                extremes=(theta[0], theta[1], radius[0], radius[1]),
                                grid_locator1=grid_locator1,
                                grid_locator2=grid_locator2,
                                tick_formatter1=tick_formatter1,
                                tick_formatter2=None,
                                )

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # adjust axis
    # the axis artist lets you call axis with
    # "bottom", "top", "left", "right"
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")

    ax1.axis["bottom"].set_visible(False)
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")

    ax1.axis["left"].label.set_text("Weight L2 norm")
    ax1.axis["top"].label.set_text("degree [\u00b0]")

    # create a parasite axes
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
    ax1.patch.zorder=0.9 # but this has a side effect that the patch is
                         # drawn twice, and possibly over some other
                         # artists. So, we decrease the zorder a bit to
                         # prevent this.

    return ax1, aux_ax


def save_figures(save_data, save_dir, args, seed):
    sns.set_theme(style="darkgrid")
 
    train_errors = save_data['train_errors']
    val_errors = save_data['val_errors']
    wt_norms = save_data['wt_norms']
    conv_norms = save_data['conv_norms']
    effective_lr = save_data['effective_lrs']
    effective_conv_lr = save_data['effective_conv_lrs']
    wt_thetas = save_data['wt_thetas']
    wt_conv_thetas = save_data['wt_conv_thetas']
    w0_thetas = save_data['w0_thetas']
    w0_conv_thetas = save_data['w0_conv_thetas']

    name= ['train_error']+['val_error']+['norm']+['conv_norm']+['wt_theta']+['wt_conv_theta']+\
        ['w0_theta']+['w0_conv_theta']+['effective_lr']+['effective_conv_lr']
    excel_data = []
    excel_data.extend([train_errors])
    excel_data.extend([val_errors])
    excel_data.extend([wt_norms])
    excel_data.extend([conv_norms])
    excel_data.extend([wt_thetas])
    excel_data.extend([wt_conv_thetas])
    excel_data.extend([w0_thetas])
    excel_data.extend([w0_conv_thetas])
    excel_data.extend([effective_lr])
    excel_data.extend([effective_conv_lr])
    avg_norm_file = pd.DataFrame(excel_data, columns=np.arange(args.epochs+1), index=[name])  
    avg_norm_file.to_excel(save_dir+'/direction_file_seed_{}_{}_{}_{}_{}.xlsx'.format(seed, args.num_sample, int(args.batch_size), args.lr, args.weight_decay))

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), train_errors, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('training error %', fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}/training_error.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), val_errors, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('validation error %', fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}/validation_error.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs+1), wt_norms, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('L2 norm', fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}/L2_norm.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs+1), conv_norms, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('L2 norm', fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}/L2_conv_norm.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), wt_thetas, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}/theta.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), wt_conv_thetas, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}/conv_theta.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), w0_thetas, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}/theta_w0.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), w0_conv_thetas, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig('{}/conv_theta_w0.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    # effective lr
    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), effective_lr, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('effecive learning rate', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.18)
    plt.savefig('{}/effective_lr.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    # effective conv lr
    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), effective_conv_lr, linewidth=3, alpha=0.9)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('effecive learning rate', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.18)
    plt.savefig('{}/effective_conv_lr.pdf'.format(save_dir), dpi=300)
    plt.cla()
    plt.close(fig)

    # weight polar plot
    # plt.clf()
    # fig = plt.figure(1, figsize=(8, 4))
    fig = plt.figure()
    # fig = plt.subplots()
    # fig.subplots_adjust(wspace=0.2, left=0.2, right=0.8)
    
    ax, aux_ax = setup_axes(fig, 111, theta=[0, 90], radius=[0, max(wt_norms)*1.1])
    # generate the data to plot
    theta=np.array([0])
    theta=np.concatenate([theta,w0_thetas]) # in degrees
    radius = wt_norms
    aux_ax.plot(theta, radius)
    # plt.tight_layout()
    fig.set_size_inches(3.75,3.75)
    fig.savefig('{}/weight_polar.pdf'.format(save_dir), dpi=300) 


# incorporated to scheduler below
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def scheduler(base_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, type='step', **kwargs):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    if type == 'step':
        decay_epoch = kwargs.setdefault('decay_epoch', [150, 225])
        lr_decay = kwargs.setdefault('lr_decay', 0.1) 

        schedule = np.array([base_value] * (epochs-warmup_epochs)*niter_per_ep) # warmup 뺀 전체 iteration임
        for nepoch in decay_epoch:
            schedule[nepoch*niter_per_ep-warmup_iters:] = schedule[nepoch*niter_per_ep-warmup_iters:] * lr_decay

    elif type == 'cosine':
        final_value = kwargs.setdefault('final_value', 0)
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

