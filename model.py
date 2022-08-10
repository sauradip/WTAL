import torch
import torch.nn as nn
import config
from transformers import CLIPTokenizer, CLIPModel
from transformers import CLIPTextModel, CLIPTextConfig



class Filter_Module(nn.Module):
    def __init__(self, len_feature):
        super(Filter_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=1,
                    stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1,
                    stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, F)        
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, 1)
        return out
        

class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )
                
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes + 1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.drop_out(out)
        out = self.conv_3(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C + 1)
        return out

class BaS_Net(nn.Module):
    def __init__(self, len_feature, num_classes, num_segments):
        super(BaS_Net, self).__init__()
        self.filter_module = Filter_Module(len_feature)
        self.len_feature = len_feature
        self.num_classes = num_classes

        self.cas_module = CAS_Module(len_feature, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.num_segments = num_segments
        self.k = num_segments // 8

        self.txt_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.bg_embeddings = nn.Parameter(
            torch.empty(1, 512)
        )

        self.proj = nn.Sequential(
            nn.Conv1d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def get_prompt(self,cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append("a video of action"+" "+c)
        return temp_prompt

    def text_features(self,vid_feat):

        B,T,C = vid_feat.size()
        cl_names = list(config.class_dict.values())
        
        act_prompt = self.get_prompt(cl_names)
        texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to('cuda')
        text_cls = self.txt_model.get_text_features(**texts) ## [cls,txt_feat] --> [200,512]
        text_emb = torch.cat([text_cls,self.bg_embeddings],dim=0).expand(B,-1,-1)  ## [bs, cls+1 ,txt_feat] --> [bs,201,512]

        return text_emb
 
    def compute_score_maps(self, visual, text):

        B,K,C = text.size()
        text_cls = text[:,:(K-1),:]
        text_cls = text_cls / text_cls.norm(dim=2, keepdim=True)
        text = text / text.norm(dim=2, keepdim=True)
        visual = torch.clamp(visual,min=1e-4)
        visual_cls = visual.mean(dim=2)
        visual = visual / visual.norm(dim=1, keepdim=True)
        visual_cls = visual_cls / visual_cls.norm(dim=1, keepdim=True)
        score_cls = torch.einsum('bc,bkc->bk', visual_cls, text_cls) * 100
        score_map = torch.einsum('bct,bkc->bkt', visual, text) * 100

        return score_map, score_cls

    def cas_module_lang(self,feat):

        feat = feat.permute(0,2,1)
        B,T,C = feat.size()
        cl_names = list(config.class_dict.values())
        # print(cl_names)
        
        act_prompt = self.get_prompt(cl_names)
        texts = self.tokenizer(act_prompt, padding=True, return_tensors="pt").to('cuda')
        text_cls = self.txt_model.get_text_features(**texts) ## [cls,txt_feat] --> [200,512]
        text_emb = torch.cat([text_cls,self.bg_embeddings],dim=0).expand(B,-1,-1)  ## [bs, cls+1 ,txt_feat] --> [bs,201,512]

        proj_feat = self.proj(feat) ## to make same dim as text
        B,K,C = text_emb.size()
        text_cls = torch.clamp(text_emb,min=1e-4)
        text_cls = text_cls / text_cls.norm(dim=2, keepdim=True)
        text = text_emb / text_emb.norm(dim=2, keepdim=True)
        visual = torch.clamp(proj_feat,min=1e-4)
        visual_cls = visual.mean(dim=2)
        visual = visual / visual.norm(dim=1, keepdim=True)
        # visual = torch.clamp(visual,min=1e-4)
        visual_cls = visual_cls / visual_cls.norm(dim=1, keepdim=True)
        # visual_cls = torch.clamp(visual_cls,min=1e-4)
        score_cls = torch.einsum('bc,bkc->bk', visual_cls, text_cls) * 100
        score_map = torch.einsum('bct,bkc->bkt', visual, text) * 100

        score_map = torch.clamp(score_map,min=1e-4)

        return score_map, score_cls


    # def forward(self, x):
    #     fore_weights = self.filter_module(x)

    #     x_supp = fore_weights * x

    #     cas_base = self.cas_module(x)
    #     cas_supp = self.cas_module(x_supp)

    #     # print(cas_base.size(),cas_supp.size()) ## torch.Size([1, T, class]) torch.Size([1, T, class])

    #     # slicing after sorting is much faster than torch.topk (https://github.com/pytorch/pytorch/issues/22812)
    #     # score_base = torch.mean(torch.topk(cas_base, self.k, dim=1)[0], dim=1)
    #     sorted_scores_base, _= cas_base.sort(descending=True, dim=1)
    #     topk_scores_base = sorted_scores_base[:, :self.k, :]
    #     score_base = torch.mean(topk_scores_base, dim=1)

    #     # score_supp = torch.mean(torch.topk(cas_supp, self.k, dim=1)[0], dim=1)
    #     sorted_scores_supp, _= cas_supp.sort(descending=True, dim=1)
    #     topk_scores_supp = sorted_scores_supp[:, :self.k, :]
    #     score_supp = torch.mean(topk_scores_supp, dim=1)

    #     score_base = self.softmax(score_base)
    #     score_supp = self.softmax(score_supp)

    #     # print(score_base.size()) ###batch x 21
    #     # print(score_supp.size())  ### batch x 21

    #     return score_base, cas_base, score_supp, cas_supp, fore_weights


    def forward(self, x):
        fore_weights = self.filter_module(x)

        x_supp = fore_weights * x

        cas_base , base_logit = self.cas_module_lang(x)
        cas_supp, supp_logit = self.cas_module_lang(x_supp)

        cas_base = cas_base.permute(0, 2, 1)
        cas_supp = cas_supp.permute(0, 2, 1)

        score_base = self.softmax(base_logit)
        score_supp = self.softmax(supp_logit)

        return score_base, cas_base, score_supp, cas_supp, fore_weights
