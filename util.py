import os, sys

GRID_FEATS_REPO_PATH = "./grid-feats-vqa"
DETECTRON2_CONFIG_FILE_PATH = os.path.join(GRID_FEATS_REPO_PATH, "configs/X-152-challenge.yaml")
DETECTRON2_MODEL_WEIGHTS_PATH = "https://dl.fbaipublicfiles.com/grid-feats-vqa/X-152pp/X-152pp.pth"

if GRID_FEATS_REPO_PATH not in sys.path:
    sys.path.append(GRID_FEATS_REPO_PATH)

import numpy as np
import os, json, cv2, random
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from grid_feats import add_attribute_config, build_detection_test_loader_with_attributes
import grid_feats

import matplotlib.pyplot as plt

class Detectron2Predictor:
    def __init__(self):
        self.cfg = get_cfg()
        add_attribute_config(self.cfg)
        self.cfg.merge_from_file(DETECTRON2_CONFIG_FILE_PATH)
        # force the final residual block to have dilations 1
        self.cfg.MODEL.RESNETS.RES5_DILATION = 1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 50
        self.cfg.MODEL.WEIGHTS = DETECTRON2_MODEL_WEIGHTS_PATH
        self.cfg.freeze()
        
        self.d2predictor = DefaultPredictor(self.cfg)

    def predict(self, original_image):
        with torch.no_grad():
            # original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            # predictions = predictor.model([inputs])[0]
            images = self.d2predictor.model.preprocess_image([inputs])
            features = self.d2predictor.model.backbone(images.tensor)
            proposals, _ = self.d2predictor.model.proposal_generator(images, features)
            # instances, _ = self.d2predictor.model.roi_heads(images, features, proposals)
            box_features = [features[f] for f in self.d2predictor.model.roi_heads.in_features]
            box_features = self.d2predictor.model.roi_heads.box_pooler(box_features, [x.proposal_boxes for x in proposals])
            box_features_ = box_features
            box_features = self.d2predictor.model.roi_heads.box_head(box_features)
            predictions = self.d2predictor.model.roi_heads.box_predictor(box_features)
            obj_labels = predictions[0].argmax(dim=1)
            attr_score = self.d2predictor.model.roi_heads.attribute_predictor(box_features, obj_labels)
            attr_score_softmax = torch.softmax(attr_score, dim=1)
            pred_instances, idx = self.d2predictor.model.roi_heads.box_predictor.inference(predictions, proposals)
            # pred_classes = pred_instances[0].pred_classes
            # pred_boxes = pred_instances[0].pred_boxes
            # pred_classes_score = pred_instances[0].scores
            
            pred_attrs_ = attr_score.argmax(dim=1)
            pred_attrs_score_ = attr_score_softmax[torch.arange(attr_score.shape[0]), pred_attrs_]
            pred_attrs = pred_attrs_[idx[0]]
            pred_attrs_score = pred_attrs_score_[idx[0]]
        
            pred_instances[0].pred_attrs = pred_attrs
            pred_instances[0].attr_scores = pred_attrs_score

        return pred_instances[0], box_features_[idx[0], :, 0, 0]

    
VISUAL_GENOME_JSON_PATH = './datasets/visual_genome/annotations/visual_genome_test.json'

class D2ResultVisualizer:
    def __init__(self):
        jt = json.load(open(VISUAL_GENOME_JSON_PATH))
        class_catalog = [''] * len(jt['categories'])
        for v in jt['categories']:
            class_catalog[v['id']] = v['name']
        attribute_catalog = [''] * len(jt['attCategories'])
        for v in jt['attCategories']:
            attribute_catalog[v['id']] = v['name']
        
        self.class_catalog = class_catalog
        self.attribute_catalog = attribute_catalog
        
    def visualize(self, image, pred_instances, attr_thresh=0.1, class_thresh=0.0):
        image = image[:, :, ::-1]
    

        plt.figure(figsize=(24, 8))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(image)
        ax.set_yticks([])
        ax.set_xticks([])
        
        ax = plt.subplot(1, 2, 2)
        ax.imshow(image)
        ax.set_yticks([])
        ax.set_xticks([])

        for i in range(len(pred_instances)):
            inst = pred_instances[i]
            if inst.scores <= class_thresh:
                continue
            bbox = inst.pred_boxes.tensor[0].cpu().numpy().astype(np.int32)
            if bbox[0] == 0: bbox[0] = 1
            if bbox[1] == 0: bbox[1] = 1
            cls = self.class_catalog[inst.pred_classes]
            if inst.attr_scores > attr_thresh:
                cls = self.attribute_catalog[inst.pred_attrs] + " " + cls
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=2, alpha=0.5))
            ax.text(bbox[0], bbox[1] - 2,
                           '%s' % (cls),
                           bbox=dict(facecolor='blue', alpha=0.5),
                           fontsize=10, color='white')

###
M2T_REPO_PATH =  "./meshed-memory-transformer"
if M2T_REPO_PATH not in sys.path:
    sys.path.append(M2T_REPO_PATH)

VOCAB_FILE_PATH = './models/vocab_m2_transformer_sc_d2.pkl'
MODEL_PARAM_PATH = './models/m2_transformer_sc_d2_best.pth'

from data import ImageDetectionsField, TextField, RawField
from models.transformer import (
    Transformer, MemoryAugmentedEncoder, 
    MeshedDecoder, ScaledDotProductAttentionMemory
)
import pickle


class M2TransformerD2:
    def __init__(self, device='cuda'):
        self.text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                                    remove_punctuation=True, nopoints=False)
        self.text_field.vocab = pickle.load(open(VOCAB_FILE_PATH, 'rb'))

        self.encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                              attention_module_kwargs={'m': 40})
        self.decoder = MeshedDecoder(len(self.text_field.vocab), 54, 3, self.text_field.vocab.stoi['<pad>'])
        self.model = Transformer(self.text_field.vocab.stoi['<bos>'], self.encoder, self.decoder)
        self.model = self.model.to(device)
        data = torch.load(MODEL_PARAM_PATH)
        self.model.load_state_dict(data['state_dict'])
        

    def predict(self, feature, max_len=40, beam_size=30, out_size=2):
        if feature.ndim < 3:
            feature = feature.unsqueeze(0)

        with torch.no_grad():
            out, _ = self.model.beam_search(
                feature, 
                max_len=max_len,
                eos_idx=self.text_field.vocab.stoi['<eos>'], 
                beam_size=beam_size, 
                out_size=out_size,
                return_probs=False,
            )
            decoded_texts = self.text_field.decode(out[0])
        if out_size == 1:
            decoded_texts = [decoded_texts]

        return decoded_texts


###
import io
import tempfile
import requests

def imread_web(url):
    res = requests.get(url)
    img = None
    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
    return img
