import os
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.scheduler import PolyLR
from utils.losses import CrossEntropy2d, FocalLoss, get_target_tensor, DiceLoss, get_target_tensor_mc
from torch.nn import BCEWithLogitsLoss
from scripts.eval import validate
from models.utils import save_model, load_model, save_da_model, load_da_model, freeze_model
import torch.nn.functional as F
from dataset.utils import source_to_target, source_to_target_np
import torch.backends.cudnn as cudnn
from dataset.utils import find_dataset_using_name
from torch.utils import data
import copy
import numpy as np
from PIL import Image
import torch.cuda.amp as amp

palette = [128, 64, 128,  # Road, 0
            244, 35, 232,  # Sidewalk, 1
            70, 70, 70,  # Building, 2
            102, 102, 156,  # Wall, 3
            190, 153, 153,  # Fence, 4
            153, 153, 153,  # pole, 5
            250, 170, 30,  # traffic light, 6
            220, 220, 0,  # traffic sign, 7
            107, 142, 35,  # vegetation, 8
            152, 251, 152,  # terrain, 9
            70, 130, 180,  # sky, 10
            220, 20, 60,  # person, 11
            255, 0, 0,  # rider, 12
            0, 0, 142,  # car, 13
            0, 0, 70,  # truck, 14
            0, 60, 100,  # bus, 15
            0, 80, 100,  # train, 16
            0, 0, 230,  # motor-bike, 17
            119, 11, 32]  # bike, 18]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def compute_ce_weights(num_classes, predictions):
    z = np.zeros((num_classes,))

    predictions = predictions.clone().detach().cpu().numpy()
    for i, (label) in enumerate(predictions):
        y = label
        # y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l

    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    return ret



def train_source_only(args,
                      model,
                      source_train_loader,
                      val_loader,
                      metrics,
                      iter_counter,
                      visualizer):
    start_iter = 0
    train_loss = 0.0
    scaler = amp.GradScaler()

    # Define optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=args.seg_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # Define scheduler
    scheduler = PolyLR(optimizer,
                       max_iters=args.max_iters,
                       power=args.power)

    # Define loss criterion
    if args.seg_loss == 'focal':
        criterion_seg = FocalLoss(num_class=args.num_classes, ignore_label=args.ignore_index)
    elif args.seg_loss == 'dice':
        criterion_seg = DiceLoss(num_classes=args.num_classes, ignore_index=args.ignore_index)
    else:
        criterion_seg = CrossEntropy2d(ignore_label=args.ignore_index)

    # Resume model if continuing training
    if args.continue_train:
        model, optimizer, scheduler, start_iter = load_model(args, model, optimizer, scheduler)

    # Start training
    source_train_loader_it = iter(source_train_loader)
    iter_counter.record_training_start(start_iter)
    for i_iter in tqdm(iter_counter.training_steps()):
        # Set model to train
        model.train()

        # Get images/labels and move them to GPUs
        try:
            images, labels, _, _ = next(source_train_loader_it)
        except:
            source_train_loader_it = iter(source_train_loader)
            images, labels, _, _ = next(source_train_loader_it)
            iter_counter.record_one_epoch()

        if args.use_st:
            mean = torch.reshape(torch.from_numpy(args.mean), (1, 3, 1, 1))
            B, C, H, W = images.shape
            mean = mean.repeat(B, 1, H, W)
            images = images - mean

        images, labels = images, labels = images.to(args.gpu_ids[0], dtype=torch.float32), labels.to(args.gpu_ids[0], dtype=torch.long)

        # Zero-grad the optimizer
        optimizer.zero_grad()

        # Train source
        with amp.autocast():
            preds, preds_sup1, preds_sup2 = model(images)
            loss1 = criterion_seg(preds, labels)
            loss2 = criterion_seg(preds_sup1, labels)
            loss3 = criterion_seg(preds_sup2, labels)
            loss_seg = loss1 + loss2 + loss3

        # Update gradients
        scaler.scale(loss_seg).backward()

        # Update model
        scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        scaler.update()

        # Update logging information
        train_loss += loss_seg.item()

        # Print losses
        if iter_counter.needs_printing():
            # Print log and visualize on tensorboard
            visualizer.info(f'Training loss at iter {iter_counter.total_steps_so_far}: {train_loss / args.print_freq}')
            visualizer.add_scalar('Training_Loss', train_loss / args.print_freq, iter_counter.total_steps_so_far)
            train_loss = 0.0

        # Validation phase
        if iter_counter.needs_validating():
            # Set model to eval
            visualizer.info('Validating model at step %d' % iter_counter.total_steps_so_far)
            validate(args, model, val_loader, metrics, visualizer, i_iter)

        # Save model
        if iter_counter.needs_saving():
            save_model(args, model, optimizer, scheduler, iter_counter)

        iter_counter.record_one_iteration()

    iter_counter.record_training_end()
    save_model(args, model, optimizer, scheduler, iter_counter)
    validate(args, model, val_loader, metrics, visualizer, i_iter)


def train_da(args, model, model_d, source_train_loader, target_train_loader, val_loader, metrics, iter_counter,
             visualizer):

    if args.ft is not None:
        model = load_model(args, model, ft=args.ft)
        validate(args, model, val_loader, metrics, visualizer, 0)
        exit()

    # Initialize variables
    cudnn.benchmark = True
    cudnn.enabled = True
    scaler = amp.GradScaler()

    start_iter = 0
    parser_source_loss = 0.0
    parser_target_loss = 0.0
    parser_d_loss = 0.0
    discriminator_source_loss = 0.0
    discriminator_target_loss = 0.0
    metrics.reset()

    # Define optimizers
    optimizer = optim.SGD(model.parameters(),
                          lr=args.seg_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    optimizer_d = optim.Adam(model_d.parameters(),
                             lr=args.d_lr,
                             betas=(args.beta1_d, args.beta2_d))

    # Define schedulers
    scheduler = PolyLR(optimizer,
                       max_iters=args.max_iters,
                       power=args.power)
    scheduler_d = PolyLR(optimizer_d,
                         max_iters=args.max_iters,
                         power=args.power)

    # Define losses criterion
    if args.seg_loss == 'focal':
        criterion_seg = FocalLoss(num_class=args.num_classes,
                                  ignore_label=args.ignore_index)
    elif args.seg_loss == 'dice':
        criterion_seg = DiceLoss(num_classes=args.num_classes, ignore_index=args.ignore_index)
    else:
        criterion_seg = CrossEntropy2d(ignore_label=args.ignore_index)
    criterion_d = BCEWithLogitsLoss()

    # Resume model if continuing training
    if args.continue_train:
        model, model_d, model_d2, optimizer, optimizer_d, \
        optimizer_d2, scheduler, scheduler_d, scheduler_d2, start_iter = load_da_model(args,
                                                                                       model,
                                                                                       model_d,
                                                                                       optimizer,
                                                                                       optimizer_d,
                                                                                       scheduler,
                                                                                       scheduler_d
                                                                                       )


    # Start training
    iter_counter.record_training_start(start_iter)
    source_train_loader_it = iter(source_train_loader)
    target_train_loader_it = iter(target_train_loader)
    visualizer.info(f'Lambda: {args.lambda_adv} at epoch: {iter_counter.total_epochs()}')
    iter_counter.record_training_start(start_iter)
    for i_iter in tqdm(iter_counter.training_steps()):
        # Set model to train
        model.train()
        model_d.train()

        # Zero-grad the optimizers
        optimizer.zero_grad()
        optimizer_d.zero_grad()

        # Get source/target images and labels and move them to GPUs
        try:
            source_images, source_labels, _, _ = next(source_train_loader_it)
        except:
            source_train_loader_it = iter(source_train_loader)
            source_images, source_labels, _, _ = next(source_train_loader_it)

        try:
            target_images, target_labels, _, name = next(target_train_loader_it)
        except:
            target_train_loader_it = iter(target_train_loader)
            target_images, target_labels, _, name = next(target_train_loader_it)
            #validate(args, model, val_loader, metrics, visualizer, i_iter)
            #model.train()

        if args.use_st:
            src_in_trg = source_to_target(source_images, target_images, L=0.01)
            mean = torch.reshape(torch.from_numpy(args.mean), (1, 3, 1, 1))
            B, C, H, W = source_images.shape
            mean = mean.repeat(B, 1, H, W)
            source_images = src_in_trg.clone() - mean
            target_images = target_images - mean

        source_images, source_labels = source_images.to(args.gpu_ids[0], dtype=torch.float32), source_labels.to(
            args.gpu_ids[0], dtype=torch.long)
        target_images, target_labels = target_images.to(args.gpu_ids[0], dtype=torch.float32), target_labels.to(
            args.gpu_ids[0], dtype=torch.long)

        # TRAIN SCENE PARSER
        # Don't accumulate gradients in discriminator
        for param in model_d.parameters():
            param.requires_grad = False

        # Train Source
        if args.model == "bisenetv2":
            with amp.autocast():
                spreds = model(source_images)
                loss_seg_source = criterion_seg(spreds, source_labels)
        else:
            with amp.autocast():
                spreds, spreds_sup1, spreds_sup2 = model(source_images)
                loss1 = criterion_seg(spreds, source_labels)
                loss2 = criterion_seg(spreds_sup1, source_labels)
                loss3 = criterion_seg(spreds_sup2, source_labels)
                loss_seg_source = loss1 + loss2 + loss3
        scaler.scale(loss_seg_source).backward()

        # Train Target
        if args.model == "bisenetv2":
            with amp.autocast():
                tpreds = model(target_images)
                loss_seg_source = criterion_seg(spreds, source_labels)
                if args.ssl == "ssl" or args.ssl == "ssl_st":
                    loss_seg_target = criterion_seg(tpreds, target_labels)
                else:
                    loss_seg_target = 0.0
        else:
            with amp.autocast():
                tpreds, tpreds_sup1, tpreds_sup2 = model(target_images)
                if args.ssl == "ssl" or args.ssl == "ssl_st":
                    losst1 = criterion_seg(tpreds, target_labels)
                    losst2 = criterion_seg(tpreds_sup1, target_labels)
                    losst3 = criterion_seg(tpreds_sup2, target_labels)
                    loss_seg_target = losst1 + losst2 + losst3
                else:
                    loss_seg_target = 0.0
        # Fool the discriminator
        with amp.autocast():
            d_output = model_d(F.softmax(tpreds, dim=1))
            loss_fool = criterion_d(d_output,
                                    get_target_tensor(d_output, "source").to(args.gpu_ids[0], dtype=torch.float))
            loss_target = loss_fool * args.lambda_adv + loss_seg_target
        scaler.scale(loss_target).backward()

        # TRAIN DISCRIMINATOR
        for param in model_d.parameters():
            param.requires_grad = True

        source_predictions = spreds.detach()
        target_predictions = tpreds.detach()

        with amp.autocast():
            d_output_source = model_d(F.softmax(source_predictions, dim=1))
            target_tensor = get_target_tensor(d_output_source, "source")
            source_d_loss = criterion_d(d_output_source, target_tensor.to(args.gpu_ids[0], dtype=torch.float)) / 2
        scaler.scale(source_d_loss).backward()

        with amp.autocast():
            d_output_target = model_d(F.softmax(target_predictions, dim=1))
            target_tensor = get_target_tensor(d_output_target, "target")
            target_d_loss = criterion_d(d_output_target, target_tensor.to(args.gpu_ids[0], dtype=torch.float)) / 2
        scaler.scale(target_d_loss).backward()

        scaler.step(optimizer)
        scaler.step(optimizer_d)
        if scheduler is not None:
            scheduler.step()
            scheduler_d.step()
        scaler.update()

        # Update logging information
        parser_source_loss += loss_seg_source.item()
        if loss_seg_target != 0.0:
            parser_target_loss += loss_seg_target.item()
        else:
            parser_target_loss += loss_seg_target
        parser_d_loss += loss_fool.item()
        discriminator_source_loss += source_d_loss.item()
        discriminator_target_loss += target_d_loss.item()

        # Print losses
        if iter_counter.needs_printing():
            # Print log and visualize on tensorboard
            visualizer.info(
                f'Parser source loss at iter {iter_counter.total_steps_so_far}: {parser_source_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Source_Loss', parser_source_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Parser target loss at iter {iter_counter.total_steps_so_far}: {parser_target_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Target_Loss', parser_target_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Parser discriminator loss at iter {iter_counter.total_steps_so_far}: {parser_d_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Discriminator_Loss', parser_d_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Discriminator Source loss at iter {iter_counter.total_steps_so_far}: {discriminator_source_loss / args.print_freq}')
            visualizer.add_scalar('Discriminator_Source_Loss', discriminator_source_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Discriminator Target loss at iter {iter_counter.total_steps_so_far}: {discriminator_target_loss / args.print_freq}')
            visualizer.add_scalar('Discriminator_Target_Loss', discriminator_target_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)

            parser_source_loss = 0.0
            parser_target_loss = 0.0
            parser_d_loss = 0.0
            discriminator_source_loss = 0.0
            discriminator_target_loss = 0.0

        # Validation phase
        if iter_counter.needs_validating():
            # Set model to eval
            visualizer.info('Validating model at step %d' % iter_counter.total_steps_so_far)
            validate(args, model, val_loader, metrics, visualizer, i_iter)

        # Save model
        if iter_counter.needs_saving():
            save_da_model(args, model, model_d, optimizer, optimizer_d, scheduler, scheduler_d, iter_counter)

        iter_counter.record_one_iteration()

    iter_counter.record_training_end()
    visualizer.info('End training')
    save_da_model(args, model, model_d, optimizer, optimizer_d, scheduler, scheduler_d, iter_counter)
    validate(args, model, val_loader, metrics, visualizer, i_iter)


def train_mcda(args, model, model_d, source_train_loader, target_train_loader, val_loader, metrics, iter_counter,
             visualizer):

    # Initialize variables
    cudnn.benchmark = True
    cudnn.enabled = True
    scaler = amp.GradScaler()

    start_iter = 0
    parser_source_loss = 0.0
    parser_target_loss = 0.0
    parser_d_loss = 0.0
    discriminator_source_loss = 0.0
    discriminator_target_loss = 0.0
    metrics.reset()

    # Define optimizers
    optimizer = optim.SGD(model.parameters(),
                          lr=args.seg_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    optimizer_d = optim.Adam(model_d.parameters(),
                             lr=args.d_lr,
                             betas=(args.beta1_d, args.beta2_d))

    # Define schedulers
    scheduler = PolyLR(optimizer,
                       max_iters=args.max_iters,
                       power=args.power)
    scheduler_d = PolyLR(optimizer_d,
                         max_iters=args.max_iters,
                         power=args.power)

    # Define losses criterion
    if args.seg_loss == 'focal':
        criterion_seg = FocalLoss(num_class=args.num_classes,
                                  ignore_label=args.ignore_index)
    else:
        criterion_seg = CrossEntropy2d(ignore_label=args.ignore_index)
    criterion_d = BCEWithLogitsLoss()

    # Resume model if continuing training
    if args.continue_train:
        model, model_d, model_d2, optimizer, optimizer_d, \
        optimizer_d2, scheduler, scheduler_d, scheduler_d2, start_iter = load_da_model(args,
                                                                                       model,
                                                                                       model_d,
                                                                                       optimizer,
                                                                                       optimizer_d,
                                                                                       scheduler,
                                                                                       scheduler_d
                                                                                       )


    # Start training
    iter_counter.record_training_start(start_iter)
    source_train_loader_it = iter(source_train_loader)
    target_train_loader_it = iter(target_train_loader)
    visualizer.info(f'Lambda: {args.lambda_adv} at epoch: {iter_counter.total_epochs()}')
    iter_counter.record_training_start(start_iter)
    for i_iter in tqdm(iter_counter.training_steps()):
        # Set model to train
        model.train()
        model_d.train()

        # Zero-grad the optimizers
        optimizer.zero_grad()
        optimizer_d.zero_grad()

        # Get source/target images and labels and move them to GPUs
        try:
            source_images, source_labels, _, _ = next(source_train_loader_it)
        except:
            source_train_loader_it = iter(source_train_loader)
            source_images, source_labels, _, _ = next(source_train_loader_it)

        try:
            target_images, target_labels, _, name = next(target_train_loader_it)
        except:
            target_train_loader_it = iter(target_train_loader)
            target_images, target_labels, _, name = next(target_train_loader_it)
            validate(args, model, val_loader, metrics, visualizer, i_iter)
            model.train()

        source_images, source_labels = source_images.to(args.gpu_ids[0], dtype=torch.float32), source_labels.to(
            args.gpu_ids[0], dtype=torch.long)
        target_images, target_labels = target_images.to(args.gpu_ids[0], dtype=torch.float32), target_labels.to(
            args.gpu_ids[0], dtype=torch.long)

        # TRAIN SCENE PARSER
        # Don't accumulate gradients in discriminator
        for param in model_d.parameters():
            param.requires_grad = False

        # Train Source
        with amp.autocast():
            spreds, spreds_sup1, spreds_sup2 = model(source_images)
            loss1 = criterion_seg(spreds, source_labels)
            loss2 = criterion_seg(spreds_sup1, source_labels)
            loss3 = criterion_seg(spreds_sup2, source_labels)
            loss_seg_source = loss1 + loss2 + loss3
        scaler.scale(loss_seg_source).backward()

        # Train Target
        with amp.autocast():
            tpreds, _, _ = model(target_images)
            # tsp = tpreds.clone().detach()
            #_, tsp = tsp.max(dim=1)
            # tsp = tsp.cpu().numpy()
            if args.use_weights:
                classes_weights = compute_ce_weights(args.num_classes, source_labels)
                classes_weights = (classes_weights - np.min(classes_weights)) / (np.max(classes_weights) - np.min(classes_weights))
                classes_weights = torch.from_numpy(classes_weights.astype(np.float32)).to(args.gpu_ids[0])
            loss_seg_target = 0.0

            # Fool the discriminator
            d_output = model_d(F.softmax(tpreds, dim=1))
            target_tensor = get_target_tensor_mc(d_output, "source").to(args.gpu_ids[0], dtype=torch.long)
            if args.use_weights:
                loss_fool = criterion_seg(d_output, target_tensor, weight=classes_weights)
            else:
                loss_fool = criterion_seg(d_output, target_tensor)
            loss_target = loss_fool * args.lambda_adv + loss_seg_target
        scaler.scale(loss_target).backward()

        # TRAIN DISCRIMINATOR
        for param in model_d.parameters():
            param.requires_grad = True

        source_predictions = spreds.detach()
        target_predictions = tpreds.detach()

        with amp.autocast():
            d_output_source = model_d(F.softmax(source_predictions, dim=1))
            target_tensor = get_target_tensor_mc(d_output_source, "source")
            source_d_loss = criterion_seg(d_output_source, target_tensor.to(args.gpu_ids[0], dtype=torch.long)) / 2
        scaler.scale(source_d_loss).backward()

        with amp.autocast():
            d_output_target = model_d(F.softmax(target_predictions, dim=1))
            target_tensor = get_target_tensor_mc(d_output_target, "target")
            target_d_loss = criterion_seg(d_output_target, target_tensor.to(args.gpu_ids[0], dtype=torch.long)) / 2
        scaler.scale(target_d_loss).backward()

        scaler.step(optimizer)
        scaler.step(optimizer_d)
        if scheduler is not None:
            scheduler.step()
            scheduler_d.step()
        scaler.update()

        # Update logging information
        parser_source_loss += loss_seg_source.item()
        if loss_seg_target != 0.0:
            parser_target_loss += loss_seg_target.item()
        else:
            parser_target_loss += loss_seg_target
        parser_d_loss += loss_fool.item()
        discriminator_source_loss += source_d_loss.item()
        discriminator_target_loss += target_d_loss.item()

        # Print losses
        if iter_counter.needs_printing():
            # Print log and visualize on tensorboard
            visualizer.info(
                f'Parser source loss at iter {iter_counter.total_steps_so_far}: {parser_source_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Source_Loss', parser_source_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Parser target loss at iter {iter_counter.total_steps_so_far}: {parser_target_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Target_Loss', parser_target_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Parser discriminator loss at iter {iter_counter.total_steps_so_far}: {parser_d_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Discriminator_Loss', parser_d_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Discriminator Source loss at iter {iter_counter.total_steps_so_far}: {discriminator_source_loss / args.print_freq}')
            visualizer.add_scalar('Discriminator_Source_Loss', discriminator_source_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Discriminator Target loss at iter {iter_counter.total_steps_so_far}: {discriminator_target_loss / args.print_freq}')
            visualizer.add_scalar('Discriminator_Target_Loss', discriminator_target_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)

            parser_source_loss = 0.0
            parser_target_loss = 0.0
            parser_d_loss = 0.0
            discriminator_source_loss = 0.0
            discriminator_target_loss = 0.0

        # Validation phase
        if iter_counter.needs_validating():
            # Set model to eval
            visualizer.info('Validating model at step %d' % iter_counter.total_steps_so_far)
            validate(args, model, val_loader, metrics, visualizer, i_iter)

        # Save model
        if iter_counter.needs_saving():
            save_da_model(args, model, model_d, optimizer, optimizer_d, scheduler, scheduler_d, iter_counter)

        iter_counter.record_one_iteration()

    iter_counter.record_training_end()
    visualizer.info('End training')
    save_da_model(args, model, model_d, optimizer, optimizer_d, scheduler, scheduler_d, iter_counter)
    validate(args, model, val_loader, metrics, visualizer, i_iter)


def train_dda(args, model, model_d, model_d_sp, source_train_loader, target_train_loader, val_loader, metrics, iter_counter,
             visualizer):

    if args.ft is not None:
        model = load_model(args, model, ft=args.ft)

    # Initialize variables
    cudnn.benchmark = True
    cudnn.enabled = True
    scaler = amp.GradScaler()

    start_iter = 0
    parser_source_loss = 0.0
    parser_target_loss = 0.0
    parser_d_loss = 0.0
    discriminator_source_loss = 0.0
    discriminator_target_loss = 0.0
    metrics.reset()

    # Define optimizers
    optimizer = optim.SGD(model.parameters(),
                          lr=args.seg_lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    optimizer_d = optim.Adam(model_d.parameters(),
                             lr=args.d_lr,
                             betas=(args.beta1_d, args.beta2_d))
    optimizer_d_sp = optim.Adam(model_d_sp.parameters(),
                               lr=args.d_lr,
                               betas=(args.beta1_d, args.beta2_d))

    # Define schedulers
    scheduler = PolyLR(optimizer,
                       max_iters=args.max_iters,
                       power=args.power)
    scheduler_d = PolyLR(optimizer_d,
                         max_iters=args.max_iters,
                         power=args.power)
    scheduler_d_sp = PolyLR(optimizer_d_sp,
                            max_iters=args.max_iters,
                            power=args.power)

    # Define losses criterion
    if args.seg_loss == 'focal':
        criterion_seg = FocalLoss(num_class=args.num_classes,
                                  ignore_label=args.ignore_index)
    elif args.seg_loss == 'dice':
        criterion_seg = DiceLoss(num_classes=args.num_classes, ignore_index=args.ignore_index)
    else:
        criterion_seg = CrossEntropy2d(ignore_label=args.ignore_index)
    criterion_d = BCEWithLogitsLoss()

    # Resume model if continuing training
    if args.continue_train:
        model, model_d, model_d_sp, optimizer, optimizer_d, \
        optimizer_d_sp, scheduler, scheduler_d, scheduler_d_sp, start_iter = load_da_model(args,
                                                                                           model,
                                                                                           model_d,
                                                                                           optimizer,
                                                                                           optimizer_d,
                                                                                           scheduler,
                                                                                           scheduler_d,
                                                                                           model_d_sp,
                                                                                           optimizer_d_sp,
                                                                                           scheduler_d_sp)


    # Start training
    iter_counter.record_training_start(start_iter)
    source_train_loader_it = iter(source_train_loader)
    target_train_loader_it = iter(target_train_loader)
    visualizer.info(f'Lambda: {args.lambda_adv} at epoch: {iter_counter.total_epochs()}')
    iter_counter.record_training_start(start_iter)
    for i_iter in tqdm(iter_counter.training_steps()):
        # Set model to train
        model.train()
        model_d.train()
        model_d_sp.train()

        # Zero-grad the optimizers
        optimizer.zero_grad()
        optimizer_d.zero_grad()
        optimizer_d_sp.zero_grad()

        # Get source/target images and labels and move them to GPUs
        try:
            source_images, source_labels, _, _ = next(source_train_loader_it)
        except:
            source_train_loader_it = iter(source_train_loader)
            source_images, source_labels, _, _ = next(source_train_loader_it)

        try:
            target_images, target_labels, _, name = next(target_train_loader_it)
        except:
            target_train_loader_it = iter(target_train_loader)
            target_images, target_labels, _, name = next(target_train_loader_it)
            validate(args, model, val_loader, metrics, visualizer, i_iter)
            model.train()

        source_images, source_labels = source_images.to(args.gpu_ids[0], dtype=torch.float32), source_labels.to(
            args.gpu_ids[0], dtype=torch.long)
        target_images, target_labels = target_images.to(args.gpu_ids[0], dtype=torch.float32), target_labels.to(
            args.gpu_ids[0], dtype=torch.long)

        # TRAIN SCENE PARSER
        # Don't accumulate gradients in discriminator
        for param in model_d.parameters():
            param.requires_grad = False
        for param in model_d_sp.parameters():
            param.requires_grad = False

        # Train Source
        with amp.autocast():
            spreds, spreds_sup1, spreds_sup2, ssp = model(source_images)
            loss1 = criterion_seg(spreds, source_labels)
            loss2 = criterion_seg(spreds_sup1, source_labels)
            loss3 = criterion_seg(spreds_sup2, source_labels)
            loss_seg_source = loss1 + loss2 + loss3
        scaler.scale(loss_seg_source).backward()

        # Train Target
        with amp.autocast():
            tpreds, _, _, tsp = model(target_images)
            loss_seg_target = 0.0
            # Fool the discriminator
            d_output = model_d(F.softmax(tpreds, dim=1))
            loss_fool = criterion_d(d_output,
                                    get_target_tensor(d_output, "source").to(args.gpu_ids[0], dtype=torch.float))

            output_sp = model_d_sp(F.softmax(tsp, dim=1))
            loss_fool_sp = criterion_d(output_sp,
                                    get_target_tensor(output_sp, "source").to(args.gpu_ids[0], dtype=torch.float))
            loss_target = loss_fool * args.lambda_adv + loss_fool_sp * args.lambda_adv + loss_seg_target
        scaler.scale(loss_target).backward()

        # TRAIN DISCRIMINATOR
        for param in model_d.parameters():
            param.requires_grad = True
        for param in model_d_sp.parameters():
            param.requires_grad = True

        source_predictions = spreds.detach()
        target_predictions = tpreds.detach()
        source_predictions_sp = ssp.detach()
        target_predictions_sp = tsp.detach()

        with amp.autocast():
            d_output_source = model_d(F.softmax(source_predictions, dim=1))
            target_tensor = get_target_tensor(d_output_source, "source")
            source_d_loss = criterion_d(d_output_source, target_tensor.to(args.gpu_ids[0], dtype=torch.float)) / 2
        scaler.scale(source_d_loss).backward()

        with amp.autocast():
            d_output_target = model_d(F.softmax(target_predictions, dim=1))
            target_tensor = get_target_tensor(d_output_target, "target")
            target_d_loss = criterion_d(d_output_target, target_tensor.to(args.gpu_ids[0], dtype=torch.float)) / 2
        scaler.scale(target_d_loss).backward()

        with amp.autocast():
            d_output_source_sp = model_d_sp(F.softmax(source_predictions_sp, dim=1))
            target_tensor_sp = get_target_tensor(d_output_source_sp, "source")
            source_d_loss_sp = criterion_d(d_output_source_sp, target_tensor_sp.to(args.gpu_ids[0], dtype=torch.float)) / 2
        scaler.scale(source_d_loss_sp).backward()

        with amp.autocast():
            d_output_target_sp = model_d_sp(F.softmax(target_predictions_sp, dim=1))
            target_tensor_sp = get_target_tensor(d_output_target_sp, "target")
            target_d_loss_sp = criterion_d(d_output_target_sp, target_tensor_sp.to(args.gpu_ids[0], dtype=torch.float)) / 2
        scaler.scale(target_d_loss_sp).backward()

        scaler.step(optimizer)
        scaler.step(optimizer_d)
        scaler.step(optimizer_d_sp)
        if scheduler is not None:
            scheduler.step()
            scheduler_d.step()
            scheduler_d_sp.step()
        scaler.update()

        # Update logging information
        parser_source_loss += loss_seg_source.item()
        if loss_seg_target != 0.0:
            parser_target_loss += loss_seg_target.item()
        else:
            parser_target_loss += loss_seg_target
        parser_d_loss += loss_fool.item()
        discriminator_source_loss += source_d_loss.item()
        discriminator_target_loss += target_d_loss.item()

        # Print losses
        if iter_counter.needs_printing():
            # Print log and visualize on tensorboard
            visualizer.info(
                f'Parser source loss at iter {iter_counter.total_steps_so_far}: {parser_source_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Source_Loss', parser_source_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Parser target loss at iter {iter_counter.total_steps_so_far}: {parser_target_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Target_Loss', parser_target_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Parser discriminator loss at iter {iter_counter.total_steps_so_far}: {parser_d_loss / args.print_freq}')
            visualizer.add_scalar('Parser_Discriminator_Loss', parser_d_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Discriminator Source loss at iter {iter_counter.total_steps_so_far}: {discriminator_source_loss / args.print_freq}')
            visualizer.add_scalar('Discriminator_Source_Loss', discriminator_source_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)
            visualizer.info(
                f'Discriminator Target loss at iter {iter_counter.total_steps_so_far}: {discriminator_target_loss / args.print_freq}')
            visualizer.add_scalar('Discriminator_Target_Loss', discriminator_target_loss / args.print_freq,
                                  iter_counter.total_steps_so_far)

            parser_source_loss = 0.0
            parser_target_loss = 0.0
            parser_d_loss = 0.0
            discriminator_source_loss = 0.0
            discriminator_target_loss = 0.0

        # Validation phase
        if iter_counter.needs_validating():
            # Set model to eval
            visualizer.info('Validating model at step %d' % iter_counter.total_steps_so_far)
            validate(args, model, val_loader, metrics, visualizer, i_iter)

        # Save model
        if iter_counter.needs_saving():
            save_da_model(args, model, model_d, optimizer, optimizer_d, scheduler, scheduler_d, iter_counter, model_d_sp, optimizer_d_sp, scheduler_d_sp)

        iter_counter.record_one_iteration()

    iter_counter.record_training_end()
    visualizer.info('End training')
    save_da_model(args, model, model_d, optimizer, optimizer_d, scheduler, scheduler_d, iter_counter, model_d_sp,
                  optimizer_d_sp, scheduler_d_sp)
    validate(args, model, val_loader, metrics, visualizer, i_iter)



def ssl(args, model, target_train_loader, visualizer):

    if args.ft is not None:
        model = load_model(args, model, ft=args.ft)

    model.eval()

    target_train_loader_it = iter(target_train_loader)
    index = 0
    for i_iter in range(len(target_train_loader)):

        target_images, _, _, target_names = next(target_train_loader_it)

        if index % 100 == 0:
            visualizer.info(f'Processed {index} images')


        image_name = []
        predicted_label = np.zeros((target_images.shape[0], 1024, 2048))
        predicted_prob = np.zeros((target_images.shape[0], 1024, 2048))

        for index, (timage, tname) in enumerate(zip(target_images, target_names)):
            if timage is not None:
                timage = timage.unsqueeze(0)
                timage = timage.to(args.gpu_ids[0], dtype=torch.float32)
                predictions = model(timage)
                target_predictions = predictions[0].unsqueeze(0)

                output = torch.nn.functional.softmax(target_predictions, dim=1)
                output = torch.nn.functional.upsample(output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
                predicted_label[index] = label.copy()
                predicted_prob[index] = prob.copy()
                image_name.append(tname)

        # sceglie le threshold guardando il batch
        thres = []
        for i in range(19):
            x = predicted_prob[predicted_label == i]
            if len(x) == 0:
                thres.append(0)
                continue
            x = np.sort(x)
            thres.append(x[np.int(np.round(len(x) * 0.5))])
        thres = np.array(thres)
        thres[thres > 0.9] = 0.9

        # crea le pseudo labels
        for idx in range(target_images.shape[0]):
            name = image_name[idx]
            label = predicted_label[idx]
            prob = predicted_prob[idx]

            # creare la label di conseguenza
            for i in range(19):
                label[(prob < thres[i]) * (label == i)] = 255

            output = np.asarray(label, dtype=np.uint8)
            mask_img = colorize_mask(output)
            output = Image.fromarray(output)
            name = name.split('/')[-1]

            os.makedirs("Cityscapes/pseudolabels", exist_ok=True)
            os.makedirs("Cityscapes/pseudolabels_rgb", exist_ok=True)

            output.save('%s/%s' % ("Cityscapes/pseudolabels", name))
            mask_img.save('%s/%s' % ("Cityscapes/pseudolabels_rgb", name))



def ssl_st(args, model, source_train_loader, target_train_loader, visualizer):

    if args.ft is not None:
        model = load_model(args, model, ft=args.ft)

    model.eval()

    source_train_loader_it = iter(source_train_loader)
    target_train_loader_it = iter(target_train_loader)
    index = 0
    for i_iter in range(len(target_train_loader)):
        try:
            source_images, source_labels, _, _ = next(source_train_loader_it)
        except:
            source_train_loader_it = iter(source_train_loader)
            source_images, source_labels, _, _ = next(source_train_loader_it)

        target_images, _, _, target_names = next(target_train_loader_it)

        if index % 100 == 0:
            visualizer.info(f'Processed {index} images')

        target_in_source = target_images.clone()
        for cnt, (trg_img, src_img) in enumerate(zip(target_images, source_images)):
            trg_in_src = torch.from_numpy(source_to_target_np(trg_img, src_img, L=0.01))
            target_in_source[cnt] = trg_in_src.clone()
            target_images[cnt] = trg_img.clone()

        mean = torch.reshape(torch.from_numpy(args.mean), (1, 3, 1, 1))
        B, C, H, W = target_images.shape
        mean = mean.repeat(B, 1, H, W)
        target_in_source = target_in_source - mean
        target_images = target_images - mean

        image_name = []
        predicted_label = np.zeros((target_images.shape[0], 1024, 2048))
        predicted_prob = np.zeros((target_images.shape[0], 1024, 2048))
        predicted_label_st = np.zeros((target_images.shape[0], 1024, 2048))
        predicted_prob_st = np.zeros((target_images.shape[0], 1024, 2048))

        for index, (timage, image_st, tname) in enumerate(zip(target_images, target_in_source, target_names)):
            if timage is not None:

                timage = timage.unsqueeze(0)
                image_st = image_st.unsqueeze(0)
                img = torch.cat((timage, image_st), dim=0)
                img = img.to(args.gpu_ids[0], dtype=torch.float32)
                predictions = model(img)
                target_predictions, target_predictions_st = predictions[0].unsqueeze(0), predictions[1].unsqueeze(0)

                output = torch.nn.functional.softmax(target_predictions, dim=1)
                output = torch.nn.functional.upsample(output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
                predicted_label[index] = label.copy()
                predicted_prob[index] = prob.copy()
                image_name.append(tname)

                output_st = torch.nn.functional.softmax(target_predictions_st, dim=1)
                output_st = torch.nn.functional.upsample(output_st, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
                output_st = output_st.transpose(1, 2, 0)
                label_st, prob_st = np.argmax(output_st, axis=2), np.max(output_st, axis=2)
                predicted_label_st[index] = label_st.copy()
                predicted_prob_st[index] = prob_st.copy()

        # sceglie le threshold guardando il batch
        thres, thres_st = [], []
        for i in range(19):
            cont, cont_st = False, False
            x = predicted_prob[predicted_label == i]
            x_st = predicted_prob_st[predicted_label_st == i]
            if len(x) == 0:
                thres.append(0)
                cont = True
            if len(x_st) == 0:
                thres_st.append(0)
                cont_st = True
            if not cont:
                x = np.sort(x)
                thres.append(x[np.int(np.round(len(x) * 0.5))])
            if not cont_st:
                x_st = np.sort(x_st)
                thres_st.append(x_st[np.int(np.round(len(x_st) * 0.5))])
        thres, thres_st = np.array(thres), np.array(thres_st)
        for cls in range(len(thres)):
            thres[cls] = max(thres[cls], thres_st[cls])
        thres[thres > 0.9] = 0.9

        # crea le pseudo labels
        for idx in range(target_images.shape[0]):
            name = image_name[idx]
            label = predicted_label[idx]
            prob = predicted_prob[idx]
            label_st = predicted_label_st[idx]
            prob_st = predicted_prob_st[idx]

            # max
            label_new = np.maximum(label, label_st)
            prob_new = np.maximum(prob, prob_st)

            # creare la label di conseguenza
            for i in range(19):
                label_new[(prob_new < thres[i]) * (label_new == i)] = 255

            output = np.asarray(label_new, dtype=np.uint8)
            mask_img = colorize_mask(output)
            output = Image.fromarray(output)
            name = name.split('/')[-1]

            os.makedirs("Cityscapes/pseudolabels_st", exist_ok=True)
            os.makedirs("Cityscapes/pseudolabels_st_rgb", exist_ok=True)

            output.save('%s/%s' % ("Cityscapes/pseudolabels_st", name))
            mask_img.save('%s/%s' % ("Cityscapes/pseudolabels_st_rgb", name))