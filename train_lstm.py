from dataset import COCOMultiLabel, categories
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import cv2
from model import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import precision_recall_fscore_support
import argparse
import os
import numpy as np
import sys
from tqdm import tqdm
from dataset import category_dict_sequential, category_dict_sequential_inv
from tensorboardX import SummaryWriter
import sys
import datetime
from munkres import Munkres

m = Munkres()

class SWA():
    """Average snapshots of a model to make the network generalize better."""
    def __init__(self, number_swa_models=0):
        """Init function."""
        self.number_swa_models = number_swa_models
        # super(SWA, self).__init__()

    def move_average(self, model, model_swa):
        """Change the weights of the SWA model."""
        self.number_swa_models += 1
        alpha = 1.0 / self.number_swa_models
        for param1, param2 in zip(model_swa.parameters(), model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha

def visualize_batch_fn(images, labels, label_lengths):
    N = images.shape[0]
    image_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3) 
    image_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    for i in range(N):
        image = images[i].data.cpu().numpy()
        image = image.transpose(1, 2, 0)
        image *= image_std
        image += image_mean
        image = (255.0 * image).astype(np.uint8)
        indexes = labels[i].data.cpu().numpy().tolist()[1:label_lengths[i].item()-1]
        indexes = [x for x in indexes]
        labels_batch = [categories[x] for x in indexes]
        cv2.imwrite("batches/%d.jpg" % i, image[:,:,::-1])
        print '%d %s' % (i, ','.join(labels_batch))
    import epdb; epdb.set_trace()

def order_the_targets_mla(scores, targets, label_lengths_sorted):
    ###
    scores_tensor = scores.clone()
    targets_tensor = targets.clone()
    ###
    device = targets.device
    scores = scores.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    targets_new = targets.copy()
    N = scores.shape[0]
    time_steps = scores.shape[1]
    indexes = np.argmax(scores, axis=2)
    changed_batch_indexes = []
    for i in range(N):
        n_labels = label_lengths_sorted[i] - 1
        current_labels = targets_tensor[i][0:n_labels]
        cost_matrix = np.zeros((n_labels, n_labels), dtype=np.float32)
        for j in range(n_labels):
            losses = -F.log_softmax(scores_tensor[i][j], dim=0)
            temp = losses[current_labels]
            cost_matrix[j, :] = temp.data.cpu().numpy()
        indexes = m.compute(cost_matrix)
        new_labels = [x[1] for x in indexes]
        current_labels = current_labels.tolist()
        new_labels = [current_labels[x] for x in new_labels]
        targets_new[i][0:n_labels] = new_labels
    targets_new = torch.LongTensor(targets_new).to(device)
    return targets_new


def order_the_targets_pla(scores, targets, label_lengths_sorted):
    device = targets.device
    scores_tensor = scores.clone()
    scores = scores.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    targets_new = targets.copy()
    targets_newest = targets.copy()
    N = scores.shape[0]
    time_steps = scores.shape[1]
    indexes = np.argmax(scores, axis=2)
    changed_batch_indexes = []
    for i in range(N):
        common_indexes = set(targets[i][0:label_lengths_sorted[i]-1]).intersection(set(indexes[i]))
        diff_indexes = set(targets[i][0:label_lengths_sorted[i]-1]).difference(set(indexes[i]))
        diff_indexes_list = list(diff_indexes)
        common_indexes_copy = common_indexes.copy()
        index_array = np.zeros((len(diff_indexes), len(diff_indexes)))
        if common_indexes != set():
            changed_batch_indexes.append(i)
            for j in range(label_lengths_sorted[i] - 1):
                if indexes[i][j] in common_indexes:
                    if indexes[i][j] != targets_new[i][j].item():
                        old_value = targets_new[i][j]
                        new_value = indexes[i][j]
                        new_value_index = np.where(
                            targets_new[i] == new_value)[0][0]
                        targets_new[i][j] = new_value
                        targets_new[i][new_value_index] = old_value
                    common_indexes.remove(indexes[i][j].item())

        targets_newest[i] = targets_new[i]
        n_different = len(diff_indexes)
        if n_different > 1:
            diff_indexes_tuples = [[count, elem]
                                   for count, elem in enumerate(
                                           targets_new[i][0:label_lengths_sorted[i]-1])
                                   if elem in diff_indexes]
            diff_indexes_locations, diff_indexes_ordered = zip(
                *diff_indexes_tuples)
            cost_matrix = np.zeros((n_different, n_different),
                                   dtype=np.float32)
            for diff_count, diff_index_location in enumerate(
                    diff_indexes_locations):
                losses = -F.log_softmax(
                    scores_tensor[i][diff_index_location], dim=0)
                temp = losses[torch.LongTensor(diff_indexes_ordered)]
                cost_matrix[diff_count, :] = temp.data.cpu().numpy()
            indexes2 = m.compute(cost_matrix)
            new_labels = [x[1] for x in indexes2]
            for new_label_count, new_label in enumerate(new_labels):
                targets_newest[i][diff_indexes_locations[new_label_count]] = diff_indexes_ordered[new_label]

    targets_newest = torch.LongTensor(targets_newest).to(device)
    return targets_newest
    
def order_the_targets(scores, targets, label_lengths_sorted):
    device = targets.device
    scores = scores.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    targets_new = targets.copy()
    N = scores.shape[0]
    time_steps = scores.shape[1]
    indexes = np.argmax(scores, axis=2)
    changed_batch_indexes = []
    for i in range(N):
        common_indexes = set(targets[i][0:label_lengths_sorted[i]-1]).intersection(set(indexes[i]))
        if common_indexes != set():
            changed_batch_indexes.append(i)
            for j in range(label_lengths_sorted[i] - 1):
                if indexes[i][j] in common_indexes:
                    if indexes[i][j] != targets_new[i][j].item():
                        old_value = targets_new[i][j]
                        new_value = indexes[i][j]
                        new_value_index = np.where(targets_new[i] == new_value)[0][0]
                        targets_new[i][j] = new_value
                        targets_new[i][new_value_index] = old_value
                    common_indexes.remove(indexes[i][j].item())

    targets_new = torch.LongTensor(targets_new).to(device)
    return targets_new

def convert_to_array(scores, targets, target_lengths):
    scores = scores.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    number_class = 80
    N = scores.shape[0]
    preds = np.zeros((N, number_class), dtype=np.float32)
    labels = np.zeros((N, number_class), dtype=np.float32)
    number_time_steps = scores.shape[1]
    for i in range(N):
        preds_image = []
        for step_t in range(number_time_steps):
            step_pred = np.argmax(scores[i][step_t])
            if category_dict_sequential_inv[step_pred] == '<end>':
                break
            preds_image.append(step_pred)
        preds[i, preds_image] = 1
        labels_image = targets[i][0:target_lengths[i]-1]
        labels[i, labels_image] = 1
    return preds, labels

def my_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

def adjust_learning_rate(optimizer, shrink_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-num_workers', default=4, type=int)
    parser.add_argument('-decoder_lr', default=1e-5, type=float)
    parser.add_argument('-encoder_lr', default=1e-5, type=float)
    parser.add_argument('-epochs', default=30, type=int)
    parser.add_argument('-snapshot', default=None)
    parser.add_argument('-hidden_size', default=512, type=int)
    parser.add_argument('-embed_size', default=256, type=int)
    parser.add_argument('-attention_size', default=512, type=int)
    parser.add_argument('-save_path', default=None)
    parser.add_argument('-test_model', action='store_true', default=False)
    parser.add_argument('-finetune_encoder', action='store_true', default=False)
    parser.add_argument('-visualize_batch', action='store_true', default=False)
    parser.add_argument('-order_free', type=str, default=None)
    parser.add_argument('-image_path',
                        help='Image path for the training and validation folders (COCO)')
    parser.add_argument('-swa_params', type=str, default='{}')
    parser.add_argument('-train_from_scratch', action='store_true', default=False)
    parser.add_argument('-encoder_weights', default=None,
                        help='weights from the encoder training')
    args = parser.parse_args()

    save_path = args.save_path
    print "Save path", save_path
    test_model = args.test_model
    if not test_model:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        log_path = os.path.join(save_path, 'logs')
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        else:
            if args.snapshot == None:
                raise ValueError('Delete the log path manually %s' % log_path)
        writer = SummaryWriter(log_dir=log_path)

    finetune_encoder = args.finetune_encoder
    if finetune_encoder:
        print "FINETUNING THE ENCODER"
    else:
        print "NOT FINETUNING"
    if test_model is True:
        assert args.snapshot is not None
    else:
        assert args.order_free in ["pla", "mla"]

    resume = 0
    highest_f1 = 0
    epochs_without_imp = 0
    iterations = 0
    encoder = Encoder(encoder_weights=args.encoder_weights)
    decoder = Decoder(args.hidden_size, args.embed_size, args.attention_size)
    encoder = encoder.to('cuda')
    decoder = decoder.to('cuda')

    snapshot = args.snapshot
    test_model = args.test_model
    train_from_scratch = args.train_from_scratch
    swa_params = eval(args.swa_params)
    finetune_encoder = args.finetune_encoder

    if not test_model:
        if finetune_encoder:
            encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    else:
        print "Testing the model"

    checkpoint = None
    if snapshot:
        checkpoint = torch.load(snapshot,  map_location=lambda storage, loc: storage)
        if (train_from_scratch and 'decoder_swa_state_dict' in checkpoint) or (test_model and 'decoder_swa_state_dict' in checkpoint):
            print "Inputting the swa weights."
            decoder.load_state_dict(checkpoint['decoder_swa_state_dict'])
            if 'encoder_swa_state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['encoder_swa_state_dict'])
            else:
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if args.test_model == False and args.train_from_scratch == False:
            resume = checkpoint['resume'] + 1
            highest_f1 = checkpoint['f1']
            iterations = checkpoint['iterations']
            epochs_without_imp = checkpoint['epochs_without_imp']
            if finetune_encoder:
                encoder_optimizer.load_state_dict(
                    checkpoint['encoder_optimizer_state_dict'])
            decoder_optimizer.load_state_dict(
                checkpoint['decoder_optimizer_state_dict'])

    if swa_params:
        from lr_scheduler import CyclicalLR
        swa_coeff = swa_params.get('swa_coeff', 0.1)
        if not args.test_model:
            scheduler_decoder = CyclicalLR(decoder_optimizer,
                                           swa_params['lr_high'],
                                           swa_params['lr_low'],
                                           swa_params['cycle_length'])
            if finetune_encoder:
                scheduler_encoder = CyclicalLR(encoder_optimizer,
                                               swa_params['lr_high'] * swa_coeff,
                                               swa_params['lr_low'] * swa_coeff,
                                               swa_params['cycle_length'])
        decoder_swa = Decoder(args.hidden_size, args.embed_size,
                              args.attention_size).to('cuda')
        encoder_swa = Encoder().to('cuda')
        print "Encoder and decoder learning rates will be overwritten"
        if checkpoint:
            decoder_swa.load_state_dict(checkpoint['decoder_swa_state_dict'])
            if 'encoder_swa_state_dict' in checkpoint:
                encoder_swa.load_state_dict(checkpoint['encoder_swa_state_dict'])
            else:
                raise ValueError("No encoder swa state dict")

            if args.train_from_scratch == False and args.test_model == False:
                iterations = checkpoint['iterations']
                number_swa_models = iterations / (swa_params['cycle_length'] + 1)
                print "# of SWA models", number_swa_models
                # iterations = 3
                swa = SWA(number_swa_models=number_swa_models)
                scheduler_decoder.curr_iter = iterations
                if finetune_encoder:
                    scheduler_encoder.curr_iter = iterations
            else:
                swa = SWA(number_swa_models=0)
                print "# of SWA models 0"
        else:
            swa = SWA(number_swa_models=0)
            print "# of SWA models 0"

    encoder.eval()
    decoder.eval()
    if swa_params:
        encoder_swa.eval()
        decoder_swa.eval()

    criterion = nn.CrossEntropyLoss()

    dataset = COCOMultiLabel(train=True,
                             classification=False,
                             image_path=args.image_path)
    dataset_val = COCOMultiLabel(train=False,
                                 classification=False,
                                 image_path=args.image_path)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False,
                            collate_fn=my_collate)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=my_collate)

    best_f1 = 0.0
    for epoch in range(resume, args.epochs):
        training = True
        if args.test_model:
            training = False

        if training:
            # train
            if finetune_encoder:
                encoder.train()
                if swa_params:
                    encoder_swa.train()
            decoder.train()
            if swa_params:
                decoder_swa.train()
            for i, batch in enumerate(dataloader):
                iterations += 1
                images = batch[0]
                labels = batch[1]
                label_lengths = batch[2]
                labels_classification = batch[3].to('cuda')
                if args.visualize_batch:
                    visualize_batch_fn(images, labels, label_lengths)
                images = images.to('cuda')
                labels = labels.to('cuda')
                label_lengths = label_lengths.to('cuda')
                encoder_out, fc_out = encoder(images)
                if swa_params:
                    if finetune_encoder:
                        encoder_swa(images)
                scores, labels_sorted, label_lengths_sorted = decoder(
                    encoder_out, fc_out, labels, label_lengths)
                # Since we decoded starting with <start>,
                # the targets are all words after <start>, up to <end>
                targets = labels_sorted[:, 1:]

                global_iter = epoch * len(dataloader) + i

                loss = 0
                # training accuracy
                if i % 50 == 0:
                    preds_train, labels_train = convert_to_array(scores, targets,
                                                                 label_lengths_sorted)
                    _, _, f1, _ = precision_recall_fscore_support(preds_train,
                                                                  labels_train,
                                                                  average='micro')
                    writer.add_scalar('train_f1', 100 * f1, global_iter)

                if args.order_free == 'pla':
                    # change the targets
                    targets = order_the_targets_pla(
                        scores, targets, label_lengths_sorted)
                elif args.order_free == 'mla':
                    targets = order_the_targets_mla(
                        scores, targets, label_lengths_sorted)   
                else:
                    raise NotImplementedError

                scores, _ = pack_padded_sequence(
                    scores, label_lengths_sorted, batch_first=True)
                targets, _ = pack_padded_sequence(
                    targets, label_lengths_sorted, batch_first=True)

                # Calculate loss
                loss_lstm = criterion(scores, targets)
                loss += loss_lstm
                if i % 50 == 0:
                    writer.add_scalar('loss', loss_lstm.item(), global_iter)
                    # learning rates
                    writer.add_scalar('decoder_lr', decoder_optimizer.param_groups[0]['lr'], global_iter)
                    if finetune_encoder:
                        writer.add_scalar('encoder_lr', encoder_optimizer.param_groups[0]['lr'], global_iter)

                decoder_optimizer.zero_grad()
                if finetune_encoder:
                    encoder_optimizer.zero_grad()
                loss.backward()
                decoder_optimizer.step()
                if finetune_encoder:
                    encoder_optimizer.step()

                if swa_params:
                    if iterations % (scheduler_decoder.cycle_length + 1) == 0:
                        swa.move_average(decoder, decoder_swa)
                        if finetune_encoder:
                            swa.move_average(encoder, encoder_swa)
                        if scheduler_decoder.print_lr()[0] != scheduler_decoder.lr_low:
                            raise AssertionError("""The learning rate is not at the lowest point.""")
                    scheduler_decoder.step()
                    if finetune_encoder:
                        scheduler_encoder.step()

                if i % 50 == 0:
                    print "epoch: %d/%d, batch: %d/%d ,loss: %.2f" % (
                        epoch, args.epochs, i, len(dataloader), loss.item())

        with torch.no_grad():
            # validation
            encoder.eval()
            decoder.eval()
            if swa_params:
                encoder_swa.eval()
                decoder_swa.eval()
            preds_all = None
            labels_all = None
            for i, batch in enumerate(tqdm(dataloader_val,
                                           total=len(dataloader_val))):
                images = batch[0]
                labels = batch[1]
                label_lengths = batch[2]
                images = images.to('cuda')
                labels = labels.to('cuda')
                label_lengths = label_lengths.to('cuda')

                if swa_params:
                    encoder_dict, fc_out = encoder_swa(images)
                    scores, labels_sorted, label_lengths_sorted = decoder_swa(
                        encoder_dict, fc_out, labels, label_lengths)
                    targets = labels_sorted[:, 1:]
                    preds, labels = convert_to_array(scores, targets,
                                                     label_lengths_sorted)

                else:
                    encoder_out, fc_out = encoder(images)
                    scores, labels_sorted, label_lengths_sorted = decoder(
                        encoder_out, fc_out, labels, label_lengths)
                    targets = labels_sorted[:, 1:]
                    preds, labels = convert_to_array(scores, targets,
                                                     label_lengths_sorted)
                        
                if i == 0:
                    preds_all = preds
                    labels_all = labels
                else:
                    preds_all = np.concatenate((preds_all, preds), axis=0)
                    labels_all = np.concatenate((labels_all, labels), axis=0)

        # this function mixes the precision and recall
        prec, recall, macro_f1, _ = precision_recall_fscore_support(preds_all,
                                                                    labels_all,
                                                                    average='macro')
        print "MACRO prec %.2f%%, recall %.2f%%, f1 %.2f%%" % (
            recall * 100, prec * 100, macro_f1 * 100)

        prec, recall, f1, _ = precision_recall_fscore_support(preds_all,
                                                              labels_all,
                                                              average='micro')
        print "MICRO prec %.2f%%, recall %.2f%%, f1 %.2f%%" % (
            recall * 100, prec * 100, f1 * 100)

        if args.test_model:
            break
        else:
            writer.add_scalar('micro_f1', f1 * 100, epoch)
            writer.add_scalar('macro_f1', macro_f1 * 100, epoch)
            save_dict = {'encoder_state_dict': encoder.state_dict(),
                         'decoder_state_dict': decoder.state_dict(),
                         'resume': epoch, 'f1': f1, 'iterations': iterations,
                         'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                         'epochs_without_imp': epochs_without_imp}
            if swa_params:
                save_dict['decoder_swa_state_dict'] = decoder_swa.state_dict()
            if finetune_encoder:
                save_dict['encoder_optimizer_state_dict'] = encoder_optimizer.state_dict()
                if swa_params:
                    save_dict['encoder_swa_state_dict'] = encoder_swa.state_dict()
            torch.save(save_dict, save_path + '/checkpoint.pth.tar')
            if f1 > highest_f1:
                print "Highest f1 score was %.2f%% now it is %.2f%%" % (highest_f1*100.0, f1*100.0)
                highest_f1 = f1
                torch.save(save_dict, save_path + "/BEST_checkpoint.pth.tar")
                epochs_without_imp = 0
            else:
                epochs_without_imp += 1
                print "Highest f1 score is still %.2f%%, epochs without imp. %d" % (
                    highest_f1*100, epochs_without_imp)
                if epochs_without_imp == 3 and swa_params == {}:
                    adjust_learning_rate(decoder_optimizer, 0.1)
                    if finetune_encoder:
                        adjust_learning_rate(encoder_optimizer, 0.1)
                    epochs_without_imp = 0
