import logging
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.experimental.optimization import fuse
import yaml
import numpy as np

from quantization_and_noise import *
from train_utils import *
import util
from model import create_model
import pickle
import os
from GA_architecture import *



def main():
    p = argparse.ArgumentParser(description='Learned Step Size Quantization')
    p.add_argument('config_file', metavar='PATH', nargs='+',default= 'test_VGGModel_config.yaml',
                   help='path to a configuration file')
    p.add_argument('--noise-scale', default=0, type=int, metavar='N',  # noise_scale交互设置接口
                   help='noise scale. ')
    p.add_argument('--final_eval_times', default=0, type=int, help='final eval times. ')

    arg = p.parse_args()
    # ************************ 训练前准备工作 ****************************
    # 获取超参数args
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml', config_file=arg.config_file)
    if arg.noise_scale != 0:
        args.quan.weight.noise_scale = float(arg.noise_scale / 100.)
        args.name = args.name + '_' + args.quan.weight.noise_method + str(args.quan.weight.noise_scale)
    if arg.final_eval_times:
        args.final_eval_times = arg.final_eval_times

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # logging文件
    log_dir = util.init_logger(args.name, output_dir, script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    if args.device.type == 'cpu' or not torch.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = torch.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        torch.cuda.set_device(args.device.gpu[0])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    # Define loss function (criterion)
    criterion = loss_func(args.model.criterion, args.device.type)
    logger.info("\nClassifier loss function : {}\n".format(criterion))
    # 初始模型的精度
    v_top1_0 = 0.
    v_top5_0 = 0.

    # Create the model
    model = create_model(args)
    model.to(args.device.type)
    if (args.model.checkpoint is not None) and (args.model.quant_checkpoint is not None):
        raise ValueError('quant_checkpoint and checkpoint can not be valued together! ')
    if args.model.checkpoint is not None:
        util.load_model_checkpoint(model, checkpoint=args.model.checkpoint,
                                   device=args.device.type, strict=args.model.strict)
        v_top1_0, v_top5_0, _ = eval(model, test_loader, criterion, -1, monitors, args)

    if args.structure.bn_fused:  # BN融合
        model.eval()
        model = fuse(model)
        logger.info("BN fused complete ................")
        v_top1_0, v_top5_0, _ = eval(model, test_loader, criterion, -1, monitors, args)
    if args.structure.bias_move:  # 去除conv/fc的bias
        model = bias_move(model)
        logger.info("bias moved complete ................")
        v_top1_0, v_top5_0, _ = eval(model, test_loader, criterion, -1, monitors, args)
    model.to('cpu')
    # func替换，增加量化层插入
    model = graph_transform(model, args.structure)
    # 量化加噪模块替换原始卷积层和全连接层
    prepare_quant_model2(model, train_loader, args.quan)
    # tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))
    logger.info('Switch normal modules to quantized modules. ')
    logger.info("***quant_model***\n" + str(model))

    model.to(args.device.type)

################################################################################################################################


    # if args.model.quant_checkpoint is not None:
    #     util.load_model_checkpoint(model, checkpoint=args.model.quant_checkpoint,
    #                                device=args.device.type, strict=args.model.strict)
    #     v_top1_0, v_top5_0, _ = eval(model, test_loader, criterion, -1, monitors, args)

    # if args.device.gpu and not args.dataloader.serialized:
    #     model = torch.nn.DataParallel(model, device_ids=args.device.gpu)   # 存在参数都在一个设备上的错误；还将影响下面读取子目录进行权重操作的过程

    # 优化器，学习率调整方式设置
    optimizer = util.get_optimizer(model, **args.optimizer)
    lr_scheduler = util.lr_scheduler(optimizer,  ##
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)

    start_epoch = 0
    N = 25  #初始种群数量 100
    m_num_acc = {}  #模型num的准确率字典---适应值

    # 获取GA的初始种群
    populations = get_initial_population(model)
    # print('populations:', populations)


    if args.resume.path:  # 继续训练
        model, start_epoch, optimizer, _ = util.load_checkpoint(
            model, args.resume.path, optimizer, args.device.type, lean=args.resume.lean)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)
    perf_scoreboard = PerformanceScoreboard(args.log.num_best_scores)



    # ************************ train & eval ****************************
    if not args.eval:  # training
        if args.resume.path or args.model.pre_trained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation val_loader)')
            top1, top5, _ = eval(model, val_loader, criterion, start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)


        # print(model)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)

            # 权重替换
            for key, items in populations.items():
                model = replace_weights(model, populations['{}'.format(key)], args)
                model.to(args.device.type)
                t_top1, t_top5, t_loss = train(model, train_loader, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args)
                # v_top1, v_top5, v_loss = eval(model, val_loader, criterion, epoch, monitors, args)  #可不要
                m_num_acc['{}'.format(key)] = t_top1

            print('before_ga:', m_num_acc)
            # logger.info('before_ga:', m_num_acc)

#####################################
            populations_after_select = selection_new(m_num_acc, N, populations)
            for i in range(N):
                # a, b = selection(3)  # 随机选择两个个体
                # child1, child2 = crossover(populations['m{}'.format(a + 1)], populations['m{}'.format(b + 1)],
                #                            0.65)  # P['ma']  P['mb']  0.65的概率进行交叉
                child1, child2, a, b = crossover_new(populations_after_select, 0.75, N)  #原0.65
                child1 = mutation(child1, 0.4)  # 变异操作  原0.5
                child1_fitness = fitness(child1, model, train_loader, criterion, optimizer, lr_scheduler, epoch, monitors, args)
                if child1_fitness > m_num_acc['m{}'.format(a + 1)]:
                    populations['m{}'.format(a + 1)] = child1
                    m_num_acc['m{}'.format(a + 1)] = child1_fitness

                print('after selection {}:'.format(i), m_num_acc)


            # tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            # tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            # tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)
            #
            # perf_scoreboard.update(v_top1, v_top5, epoch)
            # is_best = perf_scoreboard.is_best(epoch)
            # util.save_checkpoint(epoch, args.model.arch, model, optimizer, {'top1': v_top1, 'top5': v_top5}, is_best,
            #                      args.name, log_dir)

        #对适应度的列表进行排序
        max(m_num_acc, key=m_num_acc.get)
        model = replace_weights(model, populations[max(m_num_acc, key=m_num_acc.get)], args)
        model.to(args.device.type)

        logger.info('>>>>>>>> float32 model evalution')
        logger.info('==> Top1: %.3f    Top5: %.3f \n', v_top1_0, v_top5_0)
        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        v_top = []
        for t in range(args.final_eval_times):
            v_top1, v_top5, v_loss = eval(model, test_loader, criterion, -1, monitors, args)
            v_top.append([v_top1, v_top5, v_loss])
        res = np.mean(v_top, axis=0)
        res_std = np.std(v_top, axis=0)
        logger.info('==> Eval reapeat times:   %d   Avg Top1: %.3f    Top1_std: %.3f     Top5: %.3f    Loss: %.3f\n',
                    args.final_eval_times, res[0], res_std[0], res[1], res[2])


    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')


if __name__ == "__main__":
    main()




