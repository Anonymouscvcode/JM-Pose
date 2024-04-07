#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/27
    Description:
"""
import logging
import os.path as osp
import random
import time
from tqdm.auto import tqdm
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from datasets.process import get_final_preds, dark_get_final_preds
from datasets.process import get_final_preds
from datasets.transforms import reverse_transforms
# from engine.core import CORE_FUNCTION_REGISTRY, BaseFunction, AverageMeter
from engine.core.base import BaseFunction, AverageMeter
from engine.core.utils.evaluate import accuracy
# from engine.evaludate import accuracy
from engine.defaults import VAL_PHASE, TEST_PHASE, TRAIN_PHASE
from engine.defaults.constant import CORE_FUNCTION_REGISTRY
from utils.utils_bbox import cs2box
from utils.utils_folder import create_folder
from utils.utils_image_tensor import tensor2im
from datasets.process.pose_process import flip_back

@CORE_FUNCTION_REGISTRY.register()
class SubSetFunction(BaseFunction):

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if "criterion" in kwargs.keys():
            self.criterion = kwargs["criterion"]
        if "tb_log_dir" in kwargs.keys():
            self.tb_log_dir = kwargs["tb_log_dir"]
        if "writer_dict" in kwargs.keys():
            self.writer_dict = kwargs["writer_dict"]

        ##
        self.PE_Name = kwargs.get("PE_Name", "DCPOSE")
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0

    # def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, **kwargs):
    #     self.tb_writer = tb_writer_dict["writer"]
    #     self.global_steps = tb_writer_dict["global_steps"]
    #     logger = logging.getLogger(__name__)
    #     batch_time = AverageMeter()
    #     data_time = AverageMeter()
    #     losses = AverageMeter()
    #     acc_o = AverageMeter()
    #     # acc_r = AverageMeter()
    #     # switch to train mode
    #     model.train()
    #     self.use_coefficient = False
    #     self.use_KLloss = False
    #     self.new_citeration = torch.nn.KLDivLoss()
    #     self.max_iter_num = len(dataloader)
    #     self.dataloader_iter = iter(dataloader)
    #     end = time.time()
    #
    #     # for i, (input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight, meta) in enumerate(self.train_dataloader):
    #     for iter_step in range(self.max_iter_num):
    #         input_sup_start, input_sup_A, input_x, input_sup_B, input_sup_end, target_heatmaps, target_heatmaps_weight, meta = next(
    #             self.dataloader_iter)
    #         # self._before_train_iter_run(input_x, iter_step)
    #
    #         data_time.update(time.time() - end)
    #
    #         margin_left, margin_right = meta["margin_left"], meta["margin_right"]
    #         margin = torch.stack([margin_left, margin_right], dim=1).cuda()
    #         concat_input = torch.cat((input_sup_start, input_sup_A, input_x, input_sup_B, input_sup_end), 1).cuda()
    #
    #         target_heatmaps = target_heatmaps.cuda(non_blocking=True)
    #         target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)
    #         if self.use_coefficient:
    #             outputs, coefficient = model(concat_input, margin)  # hrnet, order, reverse, final
    #             loss_hrnet_init = self.criterion(outputs[0], target_heatmaps, target_heatmaps_weight)
    #             loss_hrnet = loss_hrnet_init / (2 * (coefficient[0] ** 2))
    #
    #             loss_order_init = self.criterion(outputs[1], target_heatmaps, target_heatmaps_weight)
    #             loss_order = loss_order_init / (2 * (coefficient[1] ** 2))
    #
    #             loss_reverse_init = self.criterion(outputs[2], target_heatmaps, target_heatmaps_weight)
    #             loss_reverse = loss_reverse_init / (2 * (coefficient[2] ** 2))
    #
    #             loss_final = 0.8 * self.criterion(outputs[3], target_heatmaps, target_heatmaps_weight)
    #
    #             loss = loss_hrnet + loss_order + loss_reverse + torch.log(
    #                 (coefficient[0] ** 2) * (coefficient[1] ** 2) * (coefficient[2] ** 2)) + loss_final
    #             if loss.shape[0] != 1:
    #                 loss = loss.mean()
    #         elif self.use_KLloss:
    #             outputs = model(concat_input, margin)
    #
    #             if isinstance(outputs, list) or isinstance(outputs, tuple):
    #                 loss = self.criterion(outputs[0], target_heatmaps, target_heatmaps_weight)
    #                 for pred_heatmaps in outputs[1:]:
    #                     loss += self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
    #             else:
    #                 pred_heatmaps = outputs
    #                 loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
    #             h_pred = torch.nn.functional.log_softmax(outputs[-1] * torch.unsqueeze(target_heatmaps_weight, -1))
    #             h_gt = torch.nn.functional.softmax(target_heatmaps * torch.unsqueeze(target_heatmaps_weight, -1))
    #             loss += self.new_citeration(h_pred, h_gt)
    #         else:
    #             outputs = model(concat_input, margin)
    #
    #             if isinstance(outputs, list) or isinstance(outputs, tuple):
    #                 loss = self.criterion(outputs[0], target_heatmaps, target_heatmaps_weight)
    #                 for pred_heatmaps in outputs[1:]:
    #                     loss += self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
    #             else:
    #                 pred_heatmaps = outputs
    #                 loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
    #
    #         # compute gradient and do update step
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         pred_o = outputs
    #         # pred_o = outputs[1]
    #         # pred_r = outputs[-1]
    #         # measure accuracy and record loss
    #         losses.update(loss.item(), input_x.size(0))
    #
    #         _, avg_acc_o, cnt1, _ = accuracy(pred_o.detach().cpu().numpy(),
    #                                          target_heatmaps.detach().cpu().numpy())
    #
    #         # _, avg_acc_r, cnt2, _ = accuracy(pred_r.detach().cpu().numpy(),
    #         #                                  target_heatmaps.detach().cpu().numpy())
    #         acc_o.update(avg_acc_o, cnt1)
    #         # acc_r.update(avg_acc_r, cnt2)
    #
    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()
    #         if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= self.max_iter_num - 1:
    #             # 反应的某一帧的状态
    #             # msg = 'Epoch: [{0}][{1}/{2}]\t' \
    #             #       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
    #             #       'Speed {speed:.1f} samples/s\t' \
    #             #       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
    #             #       'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
    #             #       'Accuracy_Order {acc_o.val:.3f} ({acc_o.avg:.3f})\t' \
    #             #       'Accuracy_Final {acc_r.val:.3f} ({acc_r.avg:.3f})\t'.format(epoch, iter_step, self.max_iter_num,
    #             #                                                                   batch_time=batch_time,
    #             #                                                                   speed=input_x.size(
    #             #                                                                       0) / batch_time.val,
    #             #                                                                   data_time=data_time, loss=losses,
    #             #                                                                   acc_o=acc_o, acc_r=acc_r)
    #             msg = 'Epoch: [{0}][{1}/{2}]\t' \
    #                   'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
    #                   'Speed {speed:.1f} samples/s\t' \
    #                   'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
    #                   'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
    #                   'Accuracy_Order {acc_o.val:.3f} ({acc_o.avg:.3f})\t'.format(epoch, iter_step, self.max_iter_num,
    #                                                                               batch_time=batch_time,
    #                                                                               speed=input_x.size(
    #                                                                                   0) / batch_time.val,
    #                                                                               data_time=data_time, loss=losses,
    #                                                                               acc_o=acc_o)
    #
    #             logger.info(msg)
    #
    #             # writer = self.writer_dict['writer']
    #             # global_steps = self.writer_dict['train_global_steps']
    #             # writer.add_scalar('train_loss', losses.val, global_steps)
    #             # writer.add_scalar('train_acc', acc.val, global_steps)
    #             # self.writer_dict['train_global_steps'] = global_steps + 1
    #
    #         # For Tensorboard
    #         self.tb_writer.add_scalar('train_loss', losses.val, self.global_steps)
    #         self.tb_writer.add_scalar('train_acc', acc_o.val, self.global_steps)
    #         # self.tb_writer.add_scalar('learning_rate', )
    #         self.global_steps += 1
    #
    #     tb_writer_dict["global_steps"] = self.global_steps

    def eval(self, model, dataloader, tb_writer_dict, **kwargs):
        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        acc_kf_backbone = AverageMeter()
        phase = kwargs.get("phase", VAL_PHASE)
        epoch = kwargs.get("epoch", "specified_model")
        # switch to evaluate mode
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        # prepare data fro validate
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        all_bb = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        filenames_map = {}
        filenames_counter = 0
        imgnums = []
        idx = 0
        acc_threshold = 0.7
        ##
        assert phase in [VAL_PHASE, TEST_PHASE]
        if phase == VAL_PHASE:
            FLIP_TEST = self.cfg.VAL.FLIP
            SHIFT_HEATMAP = True
        elif phase == TEST_PHASE:
            FLIP_TEST = self.cfg.TEST.FLIP
            SHIFT_HEATMAP = True
        ###
        result_output_dir, vis_output_dir = self.vis_info(logger, phase, epoch)
        ###
        logger.info(
            "PHASE:{}, FLIP_TEST:{}, SHIFT_HEATMAP:{}".format(phase, FLIP_TEST, SHIFT_HEATMAP))
        with torch.no_grad():
            end = time.time()
            num_batch = len(dataloader)
            for iter_step in tqdm(list(range(self.max_iter_num))):
                key_frame_input, sup_frame_input, target_heatmaps, target_heatmaps_weight, meta = next(
                    self.dataloader_iter)
                data_time.update(time.time() - end)
                target_heatmaps = target_heatmaps.cuda(non_blocking=True)

                pred_heatmaps, kf_bb_hm = model(key_frame_input.cuda(), sup_frame_input.cuda())
                # FLIP_TEST = True
                if FLIP_TEST:
                    input_key_flipped = key_frame_input.flip(3)
                    input_sup_flipped = sup_frame_input.flip(3)

                    pred_heatmaps_flipped, kf_bb_hm_flipped = model(input_key_flipped.cuda(),
                                                                    input_sup_flipped.cuda())

                    pred_heatmaps_flipped = flip_back(pred_heatmaps_flipped.cpu().numpy(),
                                                      dataset.flip_pairs)
                    kf_bb_hm_flipped = flip_back(kf_bb_hm_flipped.cpu().numpy(), dataset.flip_pairs)

                    pred_heatmaps_flipped = torch.from_numpy(pred_heatmaps_flipped.copy()).cuda()
                    kf_bb_hm_flipped = torch.from_numpy(kf_bb_hm_flipped.copy()).cuda()

                    if SHIFT_HEATMAP:
                        pred_heatmaps_flipped[:, :, :, 1:] = pred_heatmaps_flipped.clone()[:, :, :,
                                                             0:-1]
                        kf_bb_hm_flipped[:, :, :, 1:] = kf_bb_hm_flipped.clone()[:, :, :, 0:-1]
                    pred_heatmaps = (pred_heatmaps + pred_heatmaps_flipped) * 0.5
                    kf_bb_hm = (kf_bb_hm + kf_bb_hm_flipped) * 0.5

                _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(),
                                              target_heatmaps.detach().cpu().numpy())
                acc.update(avg_acc, cnt)

                _, kf_bb_hm_acc, cnt1, _ = accuracy(kf_bb_hm.detach().cpu().numpy(),
                                                    target_heatmaps.detach().cpu().numpy())
                acc_kf_backbone.update(kf_bb_hm_acc, cnt1)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    msg = 'Val: [{0}/{1}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(iter_step, num_batch,
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          acc=acc)
                    logger.info(msg)

                #### for eval ####
                for ff in range(len(meta['image'])):
                    cur_nm = meta['image'][ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = meta['center'].numpy()
                scale = meta['scale'].numpy()
                score = meta['score'].numpy()
                num_images = key_frame_input.size(0)

                pred_coord, our_maxvals = dark_get_final_preds(pred_heatmaps.clone().cpu().numpy(),
                                                               center, scale)
                all_preds[idx:idx + num_images, :, :2] = pred_coord
                all_preds[idx:idx + num_images, :, 2:3] = our_maxvals

                bb_coord, bb_maxvals = dark_get_final_preds(kf_bb_hm.clone().cpu().numpy(), center,
                                                            scale)
                all_bb[idx:idx + num_images, :, :2] = bb_coord
                all_bb[idx:idx + num_images, :, 2:3] = bb_maxvals

                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score

                image_path.extend(meta['image'])
                idx += num_images

                self.global_steps += 1

                self.vis_hook(meta["image"], pred_coord, our_maxvals, vis_output_dir, center, scale)

        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))
        model_name = self.cfg.MODEL.NAME

        acc_printer = PredsAccPrinter(self.cfg, all_boxes, dataset, filenames, filenames_map,
                                      imgnums, model_name,
                                      result_output_dir, self._print_name_value)
        logger.info("====> Predicting key frame heatmaps by the backbone network")
        acc_printer(all_bb)
        logger.info("====> Predicting key frame heatmaps by the local warped hm")
        acc_printer(all_preds)
        tb_writer_dict["global_steps"] = self.global_steps

    # def eval(self, model, dataloader, tb_writer_dict, **kwargs):
    #     logger = logging.getLogger(__name__)
    #
    #     self.tb_writer = tb_writer_dict["writer"]
    #     self.global_steps = tb_writer_dict["global_steps"]
    #
    #     batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    #
    #     phase = kwargs.get("phase", VAL_PHASE)
    #     epoch = kwargs.get("epoch", "specified_model")
    #     # switch to evaluate mode
    #     model.eval()
    #
    #     self.max_iter_num = len(dataloader)
    #     self.dataloader_iter = iter(dataloader)
    #     dataset = dataloader.dataset
    #     # prepare data fro validate
    #     num_samples = len(dataset)
    #     all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
    #     all_boxes = np.zeros((num_samples, 6))
    #     image_path = []
    #     filenames = []
    #     filenames_map = {}
    #     filenames_counter = 0
    #     imgnums = []
    #     idx = 0
    #     acc_threshold = 0.7
    #     person_ids = np.empty((num_samples))
    #     all_gt_keypoints = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
    #     all_initial_boxes = np.zeros((num_samples, 4))
    #     ###
    #     result_output_dir, vis_output_dir = self.vis_info(logger, phase, epoch)
    #     ###
    #
    #     with torch.no_grad():
    #         end = time.time()
    #         num_batch = len(dataloader)
    #         for iter_step in range(self.max_iter_num):
    #             data_time.update(time.time() - end)
    #             # if self.PE_Name == "MDCPOSE":
    #             if self.PE_Name == "ULDCPOSE":
    #                 input_sup_start, input_sup_A, input_x, input_sup_B, input_sup_end, target_heatmaps, target_heatmaps_weight, meta = next(
    #                     self.dataloader_iter)
    #             else:
    #                 input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight, meta = next(self.dataloader_iter)
    #
    #             # prepare model input
    #             margin_left = meta["margin_left"]
    #             margin_right = meta["margin_right"]
    #             margin = torch.stack([margin_left, margin_right], dim=1).cuda()
    #
    #             target_heatmaps = target_heatmaps.cuda(non_blocking=True)
    #
    #             if self.PE_Name == "DCPOSE":
    #                 concat_input = torch.cat((input_x, input_sup_A, input_sup_B), 1).cuda()
    #                 outputs = model(concat_input, margin=margin)
    #             elif self.PE_Name == "ULDCPOSE":
    #                 concat_input = torch.cat((input_sup_start, input_sup_A, input_x, input_sup_B, input_sup_end), 1).cuda()
    #                 outputs = model(concat_input, margin=margin)
    #
    #                 FLIP_TEST = True
    #                 SHIFT_HEATMAP = True
    #
    #                 if FLIP_TEST:
    #                     input_flipped = concat_input.flip(3)
    #                     outputs_flipped = model(input_flipped.cuda(), margin=margin)
    #
    #                     if isinstance(outputs_flipped, list):
    #                         output_flipped = outputs_flipped[-1]
    #                     else:
    #                         output_flipped = outputs_flipped
    #
    #                     output_flipped = flip_back(output_flipped.cpu().numpy(), dataset.flip_pairs)
    #                     output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
    #
    #                     # feature is not aligned, shift flipped heatmap for higher accuracy
    #                     if SHIFT_HEATMAP:
    #                         output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
    #
    #                     outputs = (outputs + output_flipped) * 0.5
    #             elif self.PE_Name == "POSEWARPER":
    #                 if random.choice(['left', 'right']) == 'left':
    #                     concat_input = torch.cat((input_x, input_sup_A), 1).cuda()
    #                 else:
    #                     concat_input = torch.cat((input_x, input_sup_B), 1).cuda()
    #
    #                 outputs = model(concat_input)
    #             else:
    #                 outputs = model(input_x.cuda())
    #
    #             # if phase == VAL_PHASE:
    #             #     if isinstance(model, torch.nn.DataParallel):
    #             #         vis_dict = getattr(model.module, "vis_dict", None)
    #             #     else:
    #             #         vis_dict = getattr(model, "vis_dict", None)
    #             #
    #             #     if vis_dict:
    #             #         self._val_iter_running(vis_dict=vis_dict, model_input=[input_x, input_sup_A, input_sup_B])
    #
    #             if isinstance(outputs, list) or isinstance(outputs, tuple):
    #                 pred_heatmaps = outputs[-1]
    #             else:
    #                 pred_heatmaps = outputs
    #
    #             _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(),
    #                                           target_heatmaps.detach().cpu().numpy())
    #             acc.update(avg_acc, cnt)
    #
    #             # measure elapsed time
    #             batch_time.update(time.time() - end)
    #             end = time.time()
    #
    #             if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
    #                 msg = 'Val: [{0}/{1}]\t' \
    #                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
    #                       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
    #                       'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(iter_step, num_batch, batch_time=batch_time,
    #                                                                       data_time=data_time, acc=acc)
    #                 logger.info(msg)
    #
    #             #### for eval ####
    #             for ff in range(len(meta['image'])):
    #                 cur_nm = meta['image'][ff]
    #                 if not cur_nm in filenames_map:
    #                     filenames_map[cur_nm] = [filenames_counter]
    #                 else:
    #                     filenames_map[cur_nm].append(filenames_counter)
    #                 filenames_counter += 1
    #
    #             center = meta['center'].numpy()
    #             scale = meta['scale'].numpy()
    #             score = meta['score'].numpy()
    #             num_images = input_x.size(0)
    #             batch_person_ids = meta['person_id'].numpy()
    #             person_ids[idx:idx + num_images] = batch_person_ids
    #
    #             box_x, box_y, box_w, box_h = meta['bbox']
    #
    #             all_initial_boxes[idx:idx + num_images, 0] = box_x
    #             all_initial_boxes[idx:idx + num_images, 1] = box_y
    #             all_initial_boxes[idx:idx + num_images, 2] = box_w
    #             all_initial_boxes[idx:idx + num_images, 3] = box_h
    #
    #             all_gt_keypoints[idx:idx + num_images, :, 0:2] = meta['joints'][:, :, 0:2]
    #             all_gt_keypoints[idx:idx + num_images, :, 2] = meta['joints_vis'][:, :, 0]
    #
    #             preds, maxvals = get_final_preds(pred_heatmaps.clone().cpu().numpy(), center, scale)
    #             all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
    #             all_preds[idx:idx + num_images, :, 2:3] = maxvals
    #             all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
    #             all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
    #             all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
    #             all_boxes[idx:idx + num_images, 5] = score
    #             image_path.extend(meta['image'])
    #             idx += num_images
    #
    #             # for vis
    #             self.vis_hook(meta["image"], preds, maxvals, vis_output_dir, center, scale)
    #
    #             # tensorboard writ
    #
    #             self.global_steps += 1
    #     logger.info('########################################')
    #     logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))
    #
    #     name_values, perf_indicator = dataset.evaluate(self.cfg, all_preds, result_output_dir, all_boxes,
    #                                                    filenames_map, filenames, imgnums,
    #                                                    person_ids=person_ids,
    #                                                    all_initial_boxes=all_initial_boxes,
    #                                                    vis_dir=vis_output_dir,
    #                                                    all_gt_keypoints=all_gt_keypoints,
    #                                                    image_path=image_path
    #                                                    )
    #
    #     model_name = self.cfg.MODEL.NAME
    #     if isinstance(name_values, list):
    #         for name_value in name_values:
    #             self._print_name_value(name_value, model_name)
    #     else:
    #         self._print_name_value(name_values, model_name)
    #
    #     tb_writer_dict["global_steps"] = self.global_steps



    def _before_train_iter_run(self, batch_x):
        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "train_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _before_val_iter_run(self, batch_x):
        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "val_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _val_iter_running(self, **kwargs):
        vis_dict = kwargs.get("vis_dict")
        #
        show_image_num = min(3, len(vis_dict["current_x"]))
        current_x = vis_dict["current_x"][0:show_image_num]  # [N,3,384,288]
        previous_x = vis_dict["previous_x"][0:show_image_num]  # [N,3,384,288]
        next_x = vis_dict["next_x"][0:show_image_num]  # [N,3,384,288]
        current_rough_heatmaps = vis_dict["current_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        previous_rough_heatmaps = vis_dict["previous_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        next_rough_heatmaps = vis_dict["next_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        diff_A = vis_dict["diff_A"][0:show_image_num]  # [N,17,96,72]
        diff_B = vis_dict["diff_B"][0:show_image_num]  # [N,17,96,72]
        diff_heatmaps = vis_dict["diff_heatmaps"][0:show_image_num]  # [N,34,96,72]
        support_heatmaps = vis_dict["support_heatmaps"][0:show_image_num]  # [N,17,96,72]
        prf_ptm_combine_featuremaps = vis_dict["prf_ptm_combine_featuremaps"][0:show_image_num]  # [N,96,96,72]
        warped_heatmaps_list = [warped_heatmaps[0:show_image_num] for warped_heatmaps in
                                vis_dict["warped_heatmaps_list"]]  # [N,17,96,72]
        output_heatmaps = vis_dict["output_heatmaps"][0:show_image_num]  # [N,17,96,72]
        ##
        # 列数代表batch的个数
        # 行数代表不同特征图的不同通道
        # show 1.

        show_three_input_image = make_grid(reverse_transforms(torch.cat([previous_x, current_x, next_x], dim=0)),
                                           nrow=show_image_num)

        self.tb_writer.add_image('{}_three_input_image'.format(self.global_steps), show_three_input_image,
                                 global_step=self.global_steps)
        # show 2.
        three_rough_heatmaps_channels = []
        current_rough_heatmap_channels = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmap_channels = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmap_channels = next_rough_heatmaps.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            three_rough_heatmaps_channels.append(current_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(previous_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(next_rough_heatmap_channels[i])

        three_heatmaps_tensor = torch.clamp_min(torch.cat(three_rough_heatmaps_channels, dim=0), 0)
        three_heatmaps_image = make_grid(three_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_three_heatmaps_image'.format(self.global_steps), three_heatmaps_image,
                                 global_step=self.global_steps)
        # show 3.
        two_diff_channels = []
        diff_A_channels = diff_A.split(1, dim=1)
        diff_B_channels = diff_B.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            two_diff_channels.append(diff_A_channels[i])
            two_diff_channels.append(diff_B_channels[i])

        two_diff_channels_tensor = torch.clamp_min(torch.cat(two_diff_channels, dim=0), 0)
        two_diff_image = make_grid(two_diff_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_two_diff_image'.format(self.global_steps), two_diff_image,
                                 global_step=self.global_steps)
        # show4.
        diff_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(diff_heatmaps, 1, dim=1), dim=0), 0)
        diff_heatmaps_channels_image = make_grid(diff_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_diff_heatmaps_channels_image'.format(self.global_steps),
                                 diff_heatmaps_channels_image,
                                 global_step=self.global_steps)
        # show5.
        support_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(support_heatmaps, 1, dim=1), dim=0), 0)
        support_heatmaps_channels_image = make_grid(support_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_support_heatmaps_channels_image'.format(self.global_steps),
                                 support_heatmaps_channels_image,
                                 global_step=self.global_steps)
        # show6.
        prf_ptm_combine_featuremaps_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(prf_ptm_combine_featuremaps, 1, dim=1), dim=0),
            0)
        prf_ptm_combine_featuremaps_channels_image = make_grid(prf_ptm_combine_featuremaps_channels_tensor,
                                                               nrow=show_image_num)
        self.tb_writer.add_image('{}_prf_ptm_combine_featuremaps_channels_image'.format(self.global_steps),
                                 prf_ptm_combine_featuremaps_channels_image, global_step=self.global_steps)
        # show7.
        warped_heatmaps_1_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[0], 1, dim=1), dim=0), 0)
        warped_heatmaps_1_channels_image = make_grid(warped_heatmaps_1_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_warped_heatmaps_1_channels_image'.format(self.global_steps),
                                 warped_heatmaps_1_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_2_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[1], 1, dim=1), dim=0), 0)
        warped_heatmaps_2_channels_image = make_grid(warped_heatmaps_2_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_warped_heatmaps_2_channels_image'.format(self.global_steps),
                                 warped_heatmaps_2_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_3_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[2], 1, dim=1), dim=0), 0)
        warped_heatmaps_3_channels_image = make_grid(warped_heatmaps_3_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_warped_heatmaps_3_channels_image'.format(self.global_steps),
                                 warped_heatmaps_3_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_4_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[3], 1, dim=1), dim=0), 0)
        warped_heatmaps_4_channels_image = make_grid(warped_heatmaps_4_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_warped_heatmaps_4_channels_image'.format(self.global_steps),
                                 warped_heatmaps_4_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_5_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[4], 1, dim=1), dim=0), 0)
        warped_heatmaps_5_channels_image = make_grid(warped_heatmaps_5_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_warped_heatmaps_5_channels_image'.format(self.global_steps),
                                 warped_heatmaps_5_channels_image,
                                 global_step=self.global_steps)

        # show8.
        output_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(output_heatmaps, 1, dim=1), dim=0), 0)
        output_heatmaps_channels_image = make_grid(output_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('{}_output_heatmaps_channels_image'.format(self.global_steps),
                                 output_heatmaps_channels_image,
                                 global_step=self.global_steps)

    def vis_info(self, logger, phase, epoch):
        if phase == TEST_PHASE:
            prefix_dir = "test"
        elif phase == TRAIN_PHASE:
            prefix_dir = "train"
        elif phase == VAL_PHASE:
            prefix_dir = "validate"
        else:
            prefix_dir = "inference"

        if isinstance(epoch, int):
            epoch = "model_{}".format(str(epoch))

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir,
                                   "use_gt_box" if self.cfg.VAL.USE_GT_BBOX else "use_precomputed_box")
        vis_output_dir = osp.join(output_dir_base, "vis")
        result_output_dir = osp.join(output_dir_base, "prediction_result")
        create_folder(vis_output_dir)
        create_folder(result_output_dir)
        logger.info("=> Vis Output Dir : {}".format(vis_output_dir))
        logger.info("=> Result Output Dir : {}".format(result_output_dir))

        if phase == VAL_PHASE:
            tensorboard_log_dir = osp.join(self.output_dir, epoch, prefix_dir, "tensorboard")
            self.tb_writer = SummaryWriter(logdir=tensorboard_log_dir)

        if self.cfg.DEBUG.VIS_SKELETON:
            logger.info("=> VIS_SKELETON")
        if self.cfg.DEBUG.VIS_BBOX:
            logger.info("=> VIS_BBOX")
        return result_output_dir, vis_output_dir

    def vis_hook(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
        cfg = self.cfg

        # prepare data
        coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
        bboxes = []
        for index in range(len(center)):
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
            from engine.core.utils.vis_helper import draw_skeleton_in_origin_image
            draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON, vis_bbox=cfg.DEBUG.VIS_BBOX)
class PredsAccPrinter(object):
    def __init__(self, cfg, all_boxes, dataset, filenames, filenames_map, imgnums, model_name,
                 result_output_dir,
                 print_name_value_func):
        self.cfg = cfg
        self.all_boxes = all_boxes
        self.dataset = dataset
        self.filenames = filenames
        self.filenames_map = filenames_map
        self.imgnums = imgnums
        self.model_name = model_name
        self.result_output_dir = result_output_dir
        self.print_name_value_func = print_name_value_func

    def __call__(self, pred_result):
        name_values, perf_indicator = self.dataset.evaluate(self.cfg, pred_result,
                                                            self.result_output_dir,
                                                            self.all_boxes, self.filenames_map,
                                                            self.filenames, self.imgnums)
        if isinstance(name_values, list):
            for name_value in name_values:
                self.print_name_value_func(name_value, self.model_name)
        else:
            self.print_name_value_func(name_values, self.model_name)

        return name_values