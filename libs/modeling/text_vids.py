from transformers import CLIPProcessor, CLIPModel
import torch
from torch import nn
import random

class FeatsProjection(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.2
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        return projected

    # def forward(self, x):
    #     projected = self.projection(x)
    #     x = self.gelu(projected)
    #     # x = self.fc(x)
    #     x = self.dropout(x)
    #     # x = x + projected
    #     # x = self.layer_norm(x)
    #     return x

class TextVidsModel(nn.Module):
    def __init__(
        self,
        text_model_name_or_path="openai/clip-vit-base-patch16", # Base-16 or Large-14
        text_embd_dim=512,
        vid_embd_dim=2048,
        projection_dim=512,
        dropout=0.2
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.text_model = CLIPModel.from_pretrained(text_model_name_or_path)
        self.txt_tokenizer = CLIPProcessor.from_pretrained(text_model_name_or_path)
        self.vid_embd_dim = vid_embd_dim
        self.vid_projection = FeatsProjection(vid_embd_dim, projection_dim, dropout) 
        self.vid_projection_layer = FeatsProjection(512, projection_dim, dropout)

        if text_model_name_or_path == "openai/clip-vit-large-patch14":
            self.vid_projection = FeatsProjection(vid_embd_dim, 768, dropout) 
            self.vid_projection_layer = FeatsProjection(512, 768, dropout)

    def get_prompt(self, cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(c)
        return temp_prompt

    def get_vid_feats(self, feats, segments):
        vid_feats_list = []
        video_feats = feats
        video_segments = segments
        
        for video_segment in video_segments:
            start = int(video_segment[0])
            end = int(video_segment[1])
            if start < 0: 
                start = 0
            elif start > video_feats.shape[1]:
                start = video_feats.shape[1] - 1
            if end == 0:
                end = 1
            if start == end:
                end = start + 1
            
            video_segment_feats = video_feats[:self.vid_embd_dim, start:end]
            # check video_segment_feats is nan and replace it's value to 0
            if torch.isnan(video_segment_feats).any():
                video_segment_feats = torch.where(torch.isnan(video_segment_feats), torch.zeros_like(video_segment_feats), video_segment_feats)

            # get means of each feature
            video_segment_feats = torch.mean(video_segment_feats, dim=1, keepdim=True)
            video_segment_feats = torch.where(torch.isnan(video_segment_feats), torch.zeros_like(video_segment_feats), video_segment_feats)

            video_segment_feats = video_segment_feats.permute(1, 0)
            vid_feats_list.append(video_segment_feats)
        
        return torch.cat(vid_feats_list, dim=0)
    
    def get_vid_feats_layer(self, feats_layer, segments, range_layer=None):
        # range_layer: [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)]
        vid_feats_list = []
        video_segments = segments
        
        for video_segment in video_segments:
            start = int(video_segment[0])
            end = int(video_segment[1])
            if start < 0: 
                start = 0

            layer = 0
            regression_rage = end - start
            if 0 <= regression_rage < 4:
                layer = 0
            elif 4 <= regression_rage < 8:
                layer = 1
            elif 8 <= regression_rage < 16:
                layer = 2
            elif 16 <= regression_rage < 32:
                layer = 3
            elif 32 <= regression_rage < 64:
                layer = 4
            elif 64 <= regression_rage < 10000:
                layer = 5

            reduce_factor = 2 ** layer
            start = start // reduce_factor
            end = end // reduce_factor

            if start > feats_layer[layer].shape[1]:
                start = feats_layer[layer].shape[1] - 1
            if end == 0:
                end = 1
            if start == end:
                end = start + 1

            video_segment_feats = feats_layer[layer][:self.vid_embd_dim, start:end]
            if torch.isnan(video_segment_feats).any():
                video_segment_feats = torch.where(torch.isnan(video_segment_feats), torch.zeros_like(video_segment_feats), video_segment_feats)

            # get means of each feature
            video_segment_feats = torch.mean(video_segment_feats, dim=1, keepdim=True)
            video_segment_feats = video_segment_feats.permute(1, 0)
            vid_feats_list.append(video_segment_feats)
        
        return torch.cat(vid_feats_list, dim=0)

    def get_txt_feats(self, labels, data_label_str):
        data_label_str_prompt = self.get_prompt(data_label_str)
        txt_labels = [data_label_str_prompt[label] for label in labels]
        
        with torch.no_grad():
            txt_labels = self.txt_tokenizer(txt_labels, padding=True, return_tensors="pt").to(self.device)
            txt_labels_feats = self.text_model.get_text_features(**txt_labels)
        return txt_labels_feats

    def txt_vid_loss(self, txt_feats, vid_feats):
        if torch.isnan(vid_feats).any():
            print("vid_feats is nan")
            import pdb; pdb.set_trace()

        logits = vid_feats @ txt_feats.T 
        targets = torch.arange(logits.shape[0]).to(self.device)
        losses = torch.nn.functional.cross_entropy(logits, targets)
        return losses

    def forward(self, video_list, data_label_dict, fpn_feats_for_text, output=None, mode='train', combine_mode=0):
        data_label_str = list(data_label_dict[0].values())
        data_label_unseen = data_label_dict[1]
        data_label_unseen_reverse = {v: k for k, v in data_label_unseen.items()}
        # get key of data_label_unseen
        data_label_unseen_str = [v for k, v in data_label_unseen.items()]
        data_label_seen = data_label_dict[2]

        if mode == 'train':
            txt_feats_list = []
            vid_feats_list = []
            vid_feats_layer_list = []
            positive_label = []

            for video_index, video in enumerate(video_list):
                # get feats in 6 layers
                vid_feats_layer_all = []
                for layer in fpn_feats_for_text:
                    vid_feats_layer_all.append(layer[video_index])

                # get text features by video labels
                txt_feats = self.get_txt_feats(video['labels'], data_label_str)
                txt_feats_list.append(txt_feats)

                # get original video features
                vid_feats = self.get_vid_feats(video['feats'], video['segments'])
                vid_feats = self.vid_projection(vid_feats)
                vid_feats_list.append(vid_feats)

                vid_feats_layer = self.get_vid_feats_layer(vid_feats_layer_all, video['segments'])
                vid_feats_layer = self.vid_projection_layer(vid_feats_layer)
                vid_feats_layer_list.append(vid_feats_layer)

                positive_label.append(video['labels'])
                
            txt_feats = torch.cat(txt_feats_list, dim=0)
            vid_feats = torch.cat(vid_feats_list, dim=0)
            vid_feats_layer_list = torch.cat(vid_feats_layer_list, dim=0)

            ### Add negative samples for open-set
            positive_label_unique = set(torch.unique(torch.cat(positive_label, dim=0)).tolist())
            negative_label = torch.tensor(list(set(data_label_seen.keys()) - positive_label_unique)).to(self.device)
            negative_txt_feats = self.get_txt_feats(negative_label, data_label_str)
            txt_feats = torch.cat([txt_feats, negative_txt_feats], dim=0)

            # import pdb; pdb.set_trace()

            ### Add negative samples for closed-set
            # positive_label_unique = set(torch.unique(torch.cat(positive_label, dim=0)).tolist())
            # negative_label = torch.tensor(list(set(data_label_unseen.keys()) - positive_label_unique)).to(self.device)
            # negative_txt_feats = self.get_txt_feats(negative_label, data_label_str)
            # txt_feats = torch.cat([txt_feats, negative_txt_feats], dim=0)

            if combine_mode == 0:
                # losses mix
                total_losses = (self.txt_vid_loss(txt_feats, vid_feats) + self.txt_vid_loss(txt_feats, vid_feats_layer_list)) / 2
            elif combine_mode == 1:
                # multi-layer only
                total_losses = self.txt_vid_loss(txt_feats, vid_feats_layer_list)
            elif combine_mode == 2:
                # original only
                total_losses = self.txt_vid_loss(txt_feats, vid_feats)

            return total_losses
        
        elif mode == 'eval':
            video_labels = []
            vid_feats_list = []
            vid_feats_layer_list = []
            positive_label = []

            for video_index, (video, out_preds) in enumerate(zip(video_list, output)):
                if len(out_preds['segments_frame']) == 0:
                    out_preds['segments_frame'] = torch.tensor([[0, 1]]).to(self.device)

                # get feats in 6 layers
                vid_feats_layer_all = []
                for layer in fpn_feats_for_text:
                    vid_feats_layer_all.append(layer[video_index])

                video_labels.append(video['labels'])

                # get video feats
                vid_feats = self.get_vid_feats(video['feats'], out_preds['segments_frame'])
                vid_feats = self.vid_projection(vid_feats)
                vid_feats_list.append(vid_feats)

                vid_feats_layer = self.get_vid_feats_layer(vid_feats_layer_all, out_preds['segments_frame'])
                vid_feats_layer = self.vid_projection_layer(vid_feats_layer)
                vid_feats_layer_list.append(vid_feats_layer)

                positive_label.append(video['labels'])

            positive_label_unique = torch.unique(torch.concat(positive_label, dim=0))
            positive_label_txt = [data_label_str[i] for i in positive_label_unique]

            ### FOR ACTIVITY NET
            ### 
            ### END: FOR ACTIVITY NET

            ### FOR THUMOS 
            final_labels = [label for label in data_label_unseen_str]
            ### END: FOR THUMOS

            vid_feats = torch.cat(vid_feats_list, dim=0)
            vid_feats_layer_list = torch.cat(vid_feats_layer_list, dim=0)
            txt_feats = self.get_txt_feats(list(range(len(final_labels))), final_labels)

            if combine_mode == 0:
                # inference mix
                logits = (torch.nn.functional.softmax(vid_feats @ txt_feats.T, dim=1) + torch.nn.functional.softmax(vid_feats_layer_list @ txt_feats.T, dim=1)) / 2
            elif combine_mode == 1:
                # Multi-layer only
                logits = torch.nn.functional.softmax(vid_feats_layer_list @ txt_feats.T, dim=1)
            elif combine_mode == 2:
                # Original only
                logits = torch.nn.functional.softmax(vid_feats @ txt_feats.T, dim=1)

            scores, preds = torch.max(logits, dim=1)
            #reverse to original key
            preds = torch.tensor([data_label_unseen_reverse[final_labels[pred]] for pred in preds]).to(self.device)

            return scores, preds