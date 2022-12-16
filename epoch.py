import segmentation_models_pytorch as smp
import sys
from tqdm import tqdm as tqdm
import os
import cv2
import torch


class Epoch:
    def __init__(
        self,
        model,
        loss,
        optimizer = None,
        s_phase = "test",
        p_dir_export = None,
        device = "cpu",
        verbose = True,
        writer = None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        if s_phase not in ["training", "validation", "test"]:
            raise ValueError(
                f'Incorrect value for s_phase: "{s_phase}"\n'
                f'Please use one of: "training", "validation", "test"'
            )
        self.s_phase = s_phase
        self.p_dir_export = p_dir_export
        self.device = device
        self.verbose = verbose
        self.writer = writer
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, image, target = None):
        if self.s_phase == "training":
            # perform inference, optimization and loss evaluation
            self.optimizer.zero_grad()
            prediction = self.model.forward(image)
            loss = self.loss(prediction, target)
            loss.backward()
            self.optimizer.step()
            return loss, prediction
        elif self.s_phase == "validation":
            # perform inference and loss evaluation
            with torch.no_grad():
                prediction = self.model.forward(image)
                loss = self.loss(prediction, target)
            return loss, prediction
        else:  # assume "test"
            # perform inference only
            prediction = self.model.forward(image)
            return None, prediction

    def on_epoch_start(self):
        if self.s_phase == "training":
            self.model.train()
        else:  # assume "validation " or "test"
            self.model.eval()

    def run(self, dataloader, i_epoch):
        self.on_epoch_start()
        logs = {}
        n_iteration_sum = 0
        l_loss_sum = 0
        n_iteration_sum = 0
        d_confusion = {
            "tp": None,
            "fp": None,
            "fn": None,
            "tn": None,
        }
        with tqdm(
            dataloader,
            desc = self.s_phase,
            file = sys.stdout,
            disable = not (self.verbose)
        ) as iterator:
            # image/target tensors have shape [n_patch, n_chan., height, width]
            # l_p_image/l_p_target tensors have shape: [n_patch] (~ lists)
            # and contain the paths to the image and target files, respectively
            for image, target, l_p_image, l_p_target in iterator:
                n_iteration_sum += 1
                image = image.to(self.device)
                if self.s_phase != "test":  # if valid target is available
                    # perform inference, with or without optimization
                    target = target.to(self.device)
                    loss, logits = self.batch_update(image, target)
                    prediction = logits.argmax(axis = 1, keepdim = True)
                    # update loss logs
                    loss_value = loss.cpu().detach().numpy()
                    l_loss_sum += loss_value
                    loss_logs = {
                        self.loss.__name__: l_loss_sum / n_iteration_sum
                    }
                    logs.update(loss_logs)
                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)
                    # update confusion matrix
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        prediction.squeeze(dim = 1),
                        target,
                        mode = 'multiclass',
                        num_classes = 19,
                        ignore_index = 19,
                    )
                    # sum statistics over image dimension and update
                    if d_confusion["tp"] is None:
                        d_confusion["tp"] = tp.sum(dim = 0, keepdim = True)
                    else:
                        d_confusion["tp"] += tp.sum(dim = 0, keepdim = True)
                    if d_confusion["fp"] is None:
                        d_confusion["fp"] = fp.sum(dim = 0, keepdim = True)
                    else:
                        d_confusion["fp"] += fp.sum(dim = 0, keepdim = True)
                    if d_confusion["fn"] is None:
                        d_confusion["fn"] = fn.sum(dim = 0, keepdim = True)
                    else:
                        d_confusion["fn"] += fn.sum(dim = 0, keepdim = True)
                    if d_confusion["tn"] is None:
                        d_confusion["tn"] = tn.sum(dim = 0, keepdim = True)
                    else:
                        d_confusion["tn"] += tn.sum(dim = 0, keepdim = True)
                else:  # in test phase, no target available
                    _, prediction = self.batch_update(image)
                # export individual predictions as images
                # (skip if no export path was given [default])
                if self.p_dir_export is None:
                    continue
                arr_prediction = prediction.detach().cpu().numpy()
                for i_image, p_target in enumerate(l_p_target):
                    # get target filename
                    fn_target = os.path.basename(p_target)
                    fn_prediction_id = fn_target.replace(
                        "_gtFine_labelIds.png",
                        "_gtFine_predictionIds.png",
                    )
                    fn_prediction_color = fn_target.replace(
                        "_gtFine_labelIds.png",
                        "_gtFine_color.png",
                    )
                    p_export_prediction_id = os.path.join(
                        self.p_dir_export,
                        fn_prediction_id,
                    )
                    p_export_prediction_color = os.path.join(
                        self.p_dir_export,
                        fn_prediction_color,
                    )
                    # convert prediction values from train_id to id
                    arr_prediction_id = dataloader.dataset.convert_trainid2id(
                        arr_prediction[i_image].transpose(1, 2, 0)
                    )
                    # convert prediction values from train_id to color
                    arr_prediction_color = dataloader.dataset.convert_trainid2color(
                        arr_prediction[i_image].transpose(1, 2, 0)
                    )
                    # convert from RGB to BGR
                    arr_prediction_color = cv2.cvtColor(
                        arr_prediction_color,
                        cv2.COLOR_RGB2BGR
                    )
                    # save prediction image
                    cv2.imwrite(p_export_prediction_id, arr_prediction_id)
                    cv2.imwrite(p_export_prediction_color, arr_prediction_color)
        # compute metrics
        if self.s_phase != "test":  # if valid target is available
            logs["iou_score"] = smp.metrics.functional.iou_score(
                tp = d_confusion["tp"],
                fp = d_confusion["fp"],
                fn = d_confusion["fn"],
                tn = d_confusion["tn"],
                reduction = 'macro-imagewise'
            ).detach().cpu().numpy()
        # write logs to Tensorboard
        if self.writer is not None:
            self.writer.add_scalar(
                f'Losses/{self.loss.__name__}',
                logs[self.loss.__name__],
                i_epoch,
            )
            self.writer.add_scalar(
                "Metrics/IoU",
                logs["iou_score"],
                i_epoch,
            )
            # TODO finish imnplementation of image logs
            # # using last computed batch for image logs
            # # (color conversion is by far not optimal, to be improved)
            # arr_prediction = prediction.detach().cpu().numpy()
            # arr_prediction = dataloader.dataset.convert_trainid2color_nchw(
            #     arr_prediction
            # )
            # self.writer.add_images(
            #     "Predictions/Color",
            #     img_tensor = torch.tensor(arr_prediction),
            #     global_step = i_epoch,
            # )
            # arr_target = target.detach().cpu().numpy()
            # arr_target = dataloader.dataset.convert_trainid2color_nchw(
            #     arr_target
            # )
            # self.writer.add_images(
            #     "Targets/Color",
            #     img_tensor = torch.tensor(arr_target),
            #     global_step = i_epoch,
            # )
            # self.writer.add_images(
            #     "Images/Color",
            #     img_tensor = image * 255,  # approximate de-normalization
            #     global_step = i_epoch,
            # )
            self.writer.flush()
        return logs
