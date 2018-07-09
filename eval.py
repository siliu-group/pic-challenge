from functools import reduce
import numpy as np
import cv2
import json
import os
import sys
from tqdm import trange


class PIC(object):
    def __init__(self, semantic_path, instance_path, relation_json, mode='gt', size=None, top_k=100):
        assert mode in ('gt', 'pred')
        self.semantic_path = semantic_path
        self.instance_path = instance_path
        self.mode = mode
        self.size = size
        self.top_k = top_k
        self.img2rels = dict()
        semantic_names = [name[:-4] for name in os.listdir(semantic_path) if name.endswith('.png')]
        instance_names = [name[:-4] for name in os.listdir(instance_path) if name.endswith('.png')]
        assert semantic_names == instance_names
        semantic_names.sort(key=str.lower)
        self.img_names = semantic_names
        img_relations = json.load(open(relation_json, 'r'))
        assert type(img_relations) == list, 'relation file format {} not supported'.format(type(img_relations))
        self.img_relations = img_relations
        self.create_index()

    def create_index(self):
        for img_relation in self.img_relations:
            if self.mode == 'gt':
                rel_numpy = np.empty((0, 3), dtype=np.int32)
                for index, rel in enumerate(img_relation['relations']):
                    temp = np.array([[rel['subject'], rel['object'], rel['relation']]], dtype=np.int32)
                    rel_numpy = np.concatenate((rel_numpy, temp), axis=0)
                self.img2rels[img_relation['name']] = rel_numpy
            elif self.mode == 'pred':
                rels = []
                scores = []
                for index, rel in enumerate(img_relation['relations']):
                    temp = np.array([[rel['subject'], rel['object'], rel['relation']]], dtype=np.int32)
                    rels.append(temp)
                    scores.append(np.array([rel['score']], dtype=np.float64))
                if len(rels) != 0:
                    rel_numpy = np.concatenate(rels, axis=0)
                    score_numpy = np.concatenate(scores, axis=0)
                    # descending sort by score
                    self.img2rels[img_relation['name']] = rel_numpy[np.argsort(-score_numpy)][:self.top_k]
                else:
                    # there is no pred_rels in this img and [-1, -1, -1] is added for convenience
                    self.img2rels[img_relation['name']] = np.array([[-1, -1, -1]], dtype=np.int32)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_semantic = cv2.imread(os.path.join(self.semantic_path, img_name+'.png'), flags=cv2.IMREAD_GRAYSCALE)
        img_instance = cv2.imread(os.path.join(self.instance_path, img_name+'.png'), flags=cv2.IMREAD_GRAYSCALE)
        if self.size is not None:
            img_semantic = cv2.resize(img_semantic, dsize=self.size, interpolation=cv2.INTER_NEAREST)
            img_instance = cv2.resize(img_instance, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        entry = {
            'semantic': img_semantic.astype(np.int32),
            'instance': img_instance.astype(np.int32),
            'relations': self.img2rels[img_name + '.jpg']
        }
        return entry

    def __len__(self):
        return len(self.img_names)


def compute_iou(target_mask, query_masks):
    N = query_masks.shape[0]
    target_masks = np.repeat(target_mask[None], N, axis=0)
    target_masks = target_masks.astype(np.int32)
    query_masks = query_masks.astype(np.int32)
    I = target_masks & query_masks
    I = I.sum(axis=2).sum(axis=1)
    U = target_masks | query_masks
    U = U.sum(axis=2).sum(axis=1) + sys.float_info.min
    return I/U


def intersect_2d(x1, x2):

    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def triplet(rels, semantic, instance):
    subs = rels[:, 0]
    objs = rels[:, 1]
    rel_cats = rels[:, 2]
    subs_mask = instance == subs[:, None, None]
    objs_mask = instance == objs[:, None, None]
    subs_semantic = subs_mask * semantic
    objs_semantic = objs_mask * semantic
    subs_class = subs_semantic.reshape(subs_semantic.shape[0], -1).max(axis=1)
    objs_class = objs_semantic.reshape(objs_semantic.shape[0], -1).max(axis=1)
    triplet_rels = np.concatenate((subs_class[:, None], objs_class[:, None], rel_cats[:, None]), axis=1)
    triplet_masks = np.concatenate((subs_mask[:, None, :, :], objs_mask[:, None, :, :]), axis=1)
    return triplet_rels, triplet_masks


def evaluate_from_dict(gt_entry, pred_entry, result_dict, iou_threshes, rel_cats, geometric_rel_cats):

    gt_rels = gt_entry['relations']
    gt_semantic = gt_entry['semantic']
    gt_instance = gt_entry['instance']
    gt_rels_nums = [0 for x in range(len(rel_cats))]
    for rel in gt_rels:
        gt_rels_nums[rel[2]-1] += 1
        if rel[2] in geometric_rel_cats.keys():
            gt_rels_nums[-2] += 1
        else:
            gt_rels_nums[-1] += 1

    pred_rels = pred_entry['relations']
    pred_semantic = pred_entry['semantic']
    pred_instance = pred_entry['instance']

    gt_triplet_rels, gt_triplet_masks = triplet(gt_rels, gt_semantic, gt_instance)
    pred_triplet_rels, pred_triplet_masks = triplet(pred_rels, pred_semantic, pred_instance)
    keeps = intersect_2d(gt_triplet_rels, pred_triplet_rels)

    gt_has_match = keeps.any(1)
    pred_to_gt = {}
    for iou_thresh in iou_threshes:
        pred_to_gt[iou_thresh] = {}
        for rel_cat_id, rel_cat_name in rel_cats.items():
            pred_to_gt[iou_thresh][rel_cat_name] = [[] for x in range(pred_rels.shape[0])]
    for gt_ind, gt_mask, keep_inds in zip(np.where(gt_has_match)[0], gt_triplet_masks[gt_has_match], keeps[gt_has_match]):
        masks = pred_triplet_masks[keep_inds]
        sub_iou = compute_iou(gt_mask[0], masks[:, 0])
        obj_iou = compute_iou(gt_mask[1], masks[:, 1])
        for iou_thresh in iou_threshes:
            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
            for i in np.where(keep_inds)[0][inds]:
                for rel_cat_id, rel_cat_name in rel_cats.items():
                    if gt_triplet_rels[int(gt_ind), 2] == rel_cat_id:
                        pred_to_gt[iou_thresh][rel_cat_name][i].append(int(gt_ind))
                if gt_triplet_rels[int(gt_ind), 2] in geometric_rel_cats.keys():
                    pred_to_gt[iou_thresh]['geometric_rel'][i].append(int(gt_ind))
                else:
                    pred_to_gt[iou_thresh]['non_geometric_rel'][i].append(int(gt_ind))
    for iou_thresh in iou_threshes:
        for rel_cat_id, rel_cat_name in rel_cats.items():
            match = reduce(np.union1d, pred_to_gt[iou_thresh][rel_cat_name])
            if gt_rels_nums[rel_cat_id-1] == 0:
                # None means this rel_cat_id does not appear in gt_rels in this img
                rec_i = None
            else:
                rec_i = float(len(match)) / float(gt_rels_nums[rel_cat_id-1])
            result_dict[iou_thresh][rel_cat_name].append(rec_i)


def main():
    size = (640, 480)
    gt_root = './demo_data/pic'
    pred_root = './demo_data/pic_pred'
    gt = PIC(semantic_path=os.path.join(gt_root, 'semantic'),
             instance_path=os.path.join(gt_root, 'instance'),
             relation_json=os.path.join(gt_root, 'relations.json'),
             mode='gt', size=size)
    pred = PIC(semantic_path=os.path.join(pred_root, 'semantic'),
               instance_path=os.path.join(pred_root, 'instance'),
               relation_json=os.path.join(pred_root, 'relations.json'),
               mode='pred', size=size, top_k=100)
    assert gt.img_names == pred.img_names
    # rel_cats are 1-30. 30: geometric_rel and 31: non_geometric_rel are added into rel_cats for convenience
    rel_cats = {
        1: 'hold', 2: 'touch', 3: 'drive', 4: 'eat', 5: 'drink', 6: 'play', 7: 'look', 8: 'throw', 9: 'ride', 10: 'talk',
        11: 'carry', 12: 'use', 13: 'pull', 14: 'push', 15: 'hit', 16: 'feed', 17: 'kick', 18: 'wear', 19: 'in-front-of', 20: 'next-to',
        21: 'on-top-of', 22: 'behind', 23: 'on', 24: 'with', 25: 'in', 26: 'sit-on', 27: 'stand-on', 28: 'lie-in', 29: 'squat', 30: 'other',
        31: 'geometric_rel', 32: 'non_geometric_rel'}
    geometric_rel_cats = {19: 'in-front-of', 20: 'next-to', 21: 'on-top-of', 22: 'behind', 23: 'on', 25: 'in'}
    iou_threshes = [0.25, 0.5, 0.75]
    # result_dict = {0.25: {'hold': [], 'touch': [], ... }, 0.5: ...}
    result_dict = {iou_thresh: {rel_cat_name: [] for rel_cat_name in rel_cats.values()} for iou_thresh in iou_threshes}
    for index in trange(len(gt)):
        evaluate_from_dict(gt[index], pred[index], result_dict, iou_threshes=iou_threshes,
                           rel_cats=rel_cats, geometric_rel_cats=geometric_rel_cats)
    for iou_thresh in iou_threshes:
        print('----------IoU: %.2f(R@100)----------' % iou_thresh)
        for rel_cat_id, rel_cat_name in rel_cats.items():
            recalls = result_dict[iou_thresh][rel_cat_name]
            while None in recalls:
                recalls.remove(None)
            if len(recalls) != 0:
                recall_mean = float('%.4f' % np.mean(recalls))
                result_dict[iou_thresh][rel_cat_name] = recall_mean
                print('%s: %.4f' % (rel_cat_name, recall_mean))
            # if all of recalls are None, it means that rel_cat_id does not appear in all imgs
            else:
                result_dict[iou_thresh][rel_cat_name] = None
                print('%s does not appear in gt_rels' % rel_cat_name)

    print('----------Final Result(R@100)----------')
    final_result_iou_25 = (result_dict[0.25]['geometric_rel'] + result_dict[0.25]['non_geometric_rel']) / 2
    final_result_iou_50 = (result_dict[0.5]['geometric_rel'] + result_dict[0.5]['non_geometric_rel']) / 2
    final_result_iou_75 = (result_dict[0.75]['geometric_rel'] + result_dict[0.75]['non_geometric_rel']) / 2
    print('IoU(0.25): %.4f' % final_result_iou_25)
    print('IoU(0.5): %.4f' % final_result_iou_50)
    print('IoU(0.75): %.4f' % final_result_iou_75)
    print('Average: %.4f' % ((final_result_iou_25 + final_result_iou_50 + final_result_iou_75) / 3))


if __name__ == '__main__':
    main()


