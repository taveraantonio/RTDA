import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data
import random
import torch.backends.cudnn as cudnn

from models.deeplabV2 import deeplabv2_resnet
from models.bisenetv1 import BiSeNet
from models.bisenetv2 import BiSeNetV2
from models.discriminator import FCDiscriminator, LightFCDiscriminator, LightLightFCDiscriminator, MultiClassLightFCDiscriminator
from models.utils import change_normalization_layer
from scripts.train import train_source_only, train_da, ssl_st, ssl, train_dda, train_mcda
from scripts.eval import test
from utils.config import Options
from utils.metrics import StreamSegMetrics
from dataset.utils import find_dataset_using_name
from utils.iter_counter import IterationCounter
from utils.visualizer import Visualizer
#from pytorch_model_summary import summary
#from fvcore.nn import FlopCountAnalysis


def main():
    # Get arguments
    args = Options().parse()

    # Set cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu_ids)

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize metric
    metrics = StreamSegMetrics(args, args.num_classes)

    # Initialize Visualizer
    visualizer = Visualizer(args)

    # Initialize Iteration counter
    iter_counter = IterationCounter(args)

    # Define/Load model
    model_d, model_dsp = None, None
    if args.model == 'deeplabv2':
        assert osp.exists(args.restore_from), f'Missing init model {args.restore_from}'
        model = deeplabv2_resnet(num_classes=args.num_classes, multi_level=args.multi_level)
        # Change normalization layer
        if args.seg_norm is not None:
            print("Changing norm with", args.seg_norm)
            change_normalization_layer(model, args.seg_norm)

        if args.is_train:
            saved_state_dict = torch.load(args.restore_from)
            if 'DeepLab_resnet_pretrained' in args.restore_from:
                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    i_parts = i.split('.')
                    if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                        new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                model.load_state_dict(new_params)
            else:
                model.load_state_dict(saved_state_dict)
    elif args.model == 'bisenetv1':
        model = BiSeNet(num_classes=args.num_classes, context_path=args.context_path)
    elif args.model == 'bisenetv2':
        model = BiSeNetV2(n_classes=args.num_classes, output_aux=False)
    else:
        raise NotImplementedError(f"Not yet supported {args.model}")
    print('Model Loaded')

    # Define discriminators
    if 'train_da' == args.train_mode:
        visualizer.info("Training with fully convolutional discriminator")
        model_d = FCDiscriminator(args.num_classes)
    elif 'train_da_light' == args.train_mode:
        visualizer.info("Training with lightweight discriminator")
        if args.use_mclight:
            model_d = MultiClassLightFCDiscriminator(num_classes=args.num_classes)
        elif args.use_light:
            model_d = LightFCDiscriminator(args.num_classes)
        elif args.use_lightlight:
            model_d = LightLightFCDiscriminator(args.num_classes)
        #if args.use_d2:
        #    model_dsp = LightFCDiscriminator(256)

    # Move model to cuda
    #print(summary(model, torch.zeros((1, 3, 512, 1024)), show_input=False))
    #print(summary(model_d, torch.zeros((1, 19, 512, 1024)), show_input=False))
    #flops = FlopCountAnalysis(model, torch.zeros((2, 3, 512, 1024)))
    #print(flops.total())

    model = torch.nn.DataParallel(model)
    model = model.to(args.gpu_ids[0])
    if model_d is not None:
        model_d = torch.nn.DataParallel(model_d)
        model_d = model_d.to(args.gpu_ids[0])
    if model_dsp is not None:
        torch.nn.DataParallel(model_dsp)
        model_dsp = model_dsp.to(args.gpu_ids[0])

    # Set cudnn
    cudnn.benchmark = True
    cudnn.enabled = True

    # Define data loaders
    source_train_loader = None
    target_train_loader = None
    val_loader = None
    test_loader = None

    mean = args.mean_prep if args.use_st else args.mean
    if args.is_train:
        # Define source train loader
        dataset_instance = find_dataset_using_name(args.source_dataset)
        source_dataset = dataset_instance(args=args,
                                          root=args.source_dataroot,
                                          mean=mean,
                                          crop_size=args.crop_size_source,
                                          train=args.is_train,
                                          ignore_index=args.ignore_index)
        source_train_loader = data.DataLoader(source_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=True,
                                              pin_memory=True)

        # Define target train loader
        dataset_instance = find_dataset_using_name(args.target_dataset)
        target_dataset = dataset_instance(root=args.target_dataroot,
                                          mean=mean,
                                          crop_size=args.crop_size_target,
                                          train=args.is_train,
                                          # max_iters=args.max_iters*args.batch_size,
                                          ignore_index=args.ignore_index,
                                          ssl=args.ssl,
                                          train_mode=args.train_mode)
        target_train_loader = data.DataLoader(target_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=True,
                                              pin_memory=True,
                                              drop_last=True)

        # Define val loader
        dataset_instance = find_dataset_using_name(args.target_dataset)
        val_dataset = dataset_instance(root=args.target_dataroot,
                                       mean=args.mean,
                                       crop_size=args.crop_size_target,
                                       train=False,
                                       ignore_index=args.ignore_index)
        val_loader = data.DataLoader(val_dataset,
                                     batch_size=args.batch_size_val,
                                     num_workers=args.num_workers,
                                     shuffle=False,
                                     pin_memory=True)
    else:
        # Define test loader
        dataset_instance = find_dataset_using_name(args.target_dataset)
        test_dataset = dataset_instance(root=args.target_dataroot,
                                        mean=args.mean,
                                        crop_size=args.crop_size_target,
                                        train=False,
                                        ignore_index=args.ignore_index)
        test_loader = data.DataLoader(test_dataset,
                                      batch_size=args.batch_size_val,
                                      num_workers=args.num_workers,
                                      shuffle=False,
                                      pin_memory=True)

    # Launch training
    if args.is_train:
        # launch train
        if 'source_only' in args.train_mode:
            train_source_only(args, model, source_train_loader, val_loader, metrics, iter_counter, visualizer)
        elif 'target_only' in args.train_mode:
            train_source_only(args, model, target_train_loader, val_loader, metrics, iter_counter, visualizer)
        elif 'train_da' in args.train_mode:
            if args.use_mclight:
                train_mcda(args, model, model_d, source_train_loader, target_train_loader, val_loader, metrics, iter_counter, visualizer)
            # elif args.use_d2:
            #     train_dda(args, model, model_d, model_dsp, source_train_loader, target_train_loader, val_loader,
            #               metrics, iter_counter,
            #              visualizer)
            else:
                train_da(args, model, model_d, source_train_loader, target_train_loader, val_loader, metrics, iter_counter, visualizer)
        elif 'ssl_st' in args.train_mode:
            ssl_st(args, model, source_train_loader, target_train_loader, visualizer)
        elif 'ssl' in args.train_mode:
            ssl(args, model, target_train_loader, visualizer)
        else:
            print("Not correct train choice")
    # Launch testing
    else:
        test(args, model, test_loader, metrics, visualizer)

    visualizer.close()


if __name__ == '__main__':
    main()

