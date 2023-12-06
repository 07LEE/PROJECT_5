import re

# CSN module definition
import torch.nn as nn
import torch.nn.functional as functional
import torch
from transformers import AutoModel


def get_nonlinear(nonlinear):
    """
    Activation function.
    """
    nonlinear_dict = {'relu': nn.ReLU(), 'tanh': nn.Tanh(
    ), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1)}
    try:
        return nonlinear_dict[nonlinear]
    except:
        raise ValueError('not a valid nonlinear type!')


class SeqPooling(nn.Module):
    """
    Sequence pooling module.

    Can do max-pooling, mean-pooling and attentive-pooling on a list of sequences of different lengths.
    """

    def __init__(self, pooling_type, hidden_dim):
        super(SeqPooling, self).__init__()
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim
        if pooling_type == 'attentive_pooling':
            self.query_vec = nn.parameter.Parameter(torch.randn(hidden_dim))

    def max_pool(self, seq):
        return seq.max(0)[0]

    def mean_pool(self, seq):
        return seq.mean(0)

    def attn_pool(self, seq):
        attn_score = torch.mm(seq, self.query_vec.view(-1, 1)).view(-1)
        attn_w = nn.Softmax(dim=0)(attn_score)
        weighted_sum = torch.mm(attn_w.view(1, -1), seq).view(-1)
        return weighted_sum

    def forward(self, batch_seq):
        pooling_fn = {'max_pooling': self.max_pool,
                      'mean_pooling': self.mean_pool,
                      'attentive_pooling': self.attn_pool}

        pooled_seq = [pooling_fn[self.pooling_type](seq) for seq in batch_seq]
        return torch.stack(pooled_seq, dim=0)


class MLP_Scorer(nn.Module):
    """
    MLP scorer module.

    A perceptron with two layers.
    """

    def __init__(self, args, classifier_input_size):
        super(MLP_Scorer, self).__init__()
        self.scorer = nn.ModuleList()

        self.scorer.append(nn.Linear(classifier_input_size,
                           args.classifier_intermediate_dim))
        self.scorer.append(nn.Linear(args.classifier_intermediate_dim, 1))
        self.nonlinear = get_nonlinear(args.nonlinear_type)

    def forward(self, x):
        for model in self.scorer:
            x = self.nonlinear(model(x))
        return x


class CSN(nn.Module):
    """
    Candidate Scoring Network.

    It's built on BERT with an MLP and other simple components.
    """

    def __init__(self, args):
        super(CSN, self).__init__()
        self.args = args
        self.bert_model = AutoModel.from_pretrained(args.bert_pretrained_dir)
        self.pooling = SeqPooling(
            args.pooling_type, self.bert_model.config.hidden_size)
        self.mlp_scorer = MLP_Scorer(
            args, self.bert_model.config.hidden_size * 3)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, features, sent_char_lens, mention_poses, quote_idxes, true_index, device):
        """
        params
            features: the candidate-specific segments (CSS) converted into the form of BERT input.  
            sent_char_lens: character-level lengths of sentences in CSSs.
                [[character-level length of sentence 1,...] in the CSS of candidate 1,...]
            mention_poses: the positions of the nearest candidate mentions.
                [(sentence-level index of nearest mention in CSS, 
                 character-level index of the leftmost character of nearest mention in CSS, 
                 character-level index of the rightmost character + 1) of candidate 1,...]
            quote_idxes: the sentence-level index of the quotes in CSSs.
                [index of quote in the CSS of candidate 1,...]
            true_index: the index of the true speaker.
            device: gpu/tpu/cpu device.
        """
        # encoding
        qs_hid = []
        ctx_hid = []
        cdd_hid = []

        for i, (cdd_sent_char_lens, cdd_mention_pos, cdd_quote_idx) in enumerate(zip(sent_char_lens, mention_poses, quote_idxes)):

            bert_output = self.bert_model(torch.tensor([features[i].input_ids], dtype=torch.long).to(device), token_type_ids=None,
                                          attention_mask=torch.tensor([features[i].input_mask], dtype=torch.long).to(device))

            accum_char_len = [0]

            for sent_idx, char_len in enumerate(cdd_sent_char_lens):
                accum_char_len.append(accum_char_len[-1] + char_len)

            CSS_hid = bert_output['last_hidden_state'][0][1:sum(
                cdd_sent_char_lens) + 1]
            qs_hid.append(
                CSS_hid[accum_char_len[cdd_quote_idx]:accum_char_len[cdd_quote_idx + 1]])

            if len(cdd_sent_char_lens) == 1:
                ctx_hid.append(torch.zeros(1, CSS_hid.size(1)).to(device))
            elif cdd_mention_pos[0] == 0:
                ctx_hid.append(CSS_hid[:accum_char_len[-2]])
            else:
                ctx_hid.append(CSS_hid[accum_char_len[1]:])

            cdd_hid.append(CSS_hid[cdd_mention_pos[1]:cdd_mention_pos[2]])

        # pooling
        qs_rep = self.pooling(qs_hid)
        ctx_rep = self.pooling(ctx_hid)
        cdd_rep = self.pooling(cdd_hid)

        # concatenate
        feature_vector = torch.cat([qs_rep, ctx_rep, cdd_rep], dim=-1)

        # dropout
        feature_vector = self.dropout(feature_vector)

        # scoring
        scores = self.mlp_scorer(feature_vector).view(-1)
        scores_false = [scores[i]
                        for i in range(scores.size(0)) if i != true_index]
        scores_true = [scores[true_index] for i in range(scores.size(0) - 1)]

        return scores, scores_false, scores_true


class KCSN(nn.Module):
    """
    Candidate Scoring Network.

    It's built on BERT with an MLP and other simple components.
    """
    def __init__(self, args):
        super(KCSN, self).__init__()
        self.args = args
        self.bert_model = AutoModel.from_pretrained(args.bert_pretrained_dir)
        self.pooling = SeqPooling(
            args.pooling_type, self.bert_model.config.hidden_size)
        self.mlp_scorer = MLP_Scorer(
            args, self.bert_model.config.hidden_size * 3)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, features, sent_char_lens, mention_poses, quote_idxes, true_index, device, tokens_list, cut_css):
        # encoding
        qs_hid = []
        ctx_hid = []
        cdd_hid = []

        unk_loc_li = []
        unk_loc = 0

        for i, (cdd_sent_char_lens, cdd_mention_pos, cdd_quote_idx) in enumerate(zip(sent_char_lens, mention_poses, quote_idxes)):
            unk_loc += 1
            bert_output = self.bert_model(torch.tensor([features[i].input_ids], dtype=torch.long).to(device), token_type_ids=None,
                                          attention_mask=torch.tensor([features[i].input_mask], dtype=torch.long).to(device))

            modified_list = [s.replace('#', '') for s in tokens_list[i]]

            cnt = 1
            verify = 0
            verify_name = 0
            num_check = 0
            num_vid = -999
            accum_char_len = [0]

            for idx, txt in enumerate(cut_css[i]):
                result_string = ''.join(txt)
                replace_dict = {']': r'\]', '[': r'\[',
                                '?': r'\?', '-': r'\-', '!': r'\!'}
                string_processing = result_string[-7:].translate(
                    str.maketrans(replace_dict))

                pattern = re.compile(rf'[{string_processing}]')

                cnt = 1

                if num_check == 1000:
                    accum_char_len.append(num_vid)

                num_check = 1000
                for string in modified_list:
                    string_nospace = string.replace(' ', '')
                    if len(accum_char_len) > idx + 1:
                        continue

                    for letter in string_nospace:
                        match_result = pattern.match(letter)
                        if match_result:
                            end_index = match_result.end()
                            verify += 1
                            if verify == len(result_string[-7:]):
                                if cnt > accum_char_len[-1]:
                                    accum_char_len.append(cnt)
                                verify = 0
                                num_check = len(accum_char_len)
                        else:
                            verify = 0
                    cnt += 1

            # 빈 부분 해결
            if num_check == 1000:
                accum_char_len.append(num_vid)

            if -999 in accum_char_len:
                unk_loc_li.append(unk_loc)
                continue

            CSS_hid = bert_output['last_hidden_state'][0][1:sum(cdd_sent_char_lens) + 1]
            qs_hid.append(CSS_hid[accum_char_len[cdd_quote_idx]:accum_char_len[cdd_quote_idx + 1]])

            # 발화자 부분 찾아서 - bert tokenizer 된 부분을 인덱싱 하는 부분
            cnt = 1
            cdd_mention_pos_bert_li = []
            cdd_mention_pos_unk = []
            name = cut_css[i][cdd_mention_pos[0]][cdd_mention_pos[3]]

            pattern_name = re.compile(rf'[{name}]')
            pattern_unk = re.compile(r'[\[UNK\]]')

            if len(accum_char_len) < cdd_mention_pos[0]+1:
                maxx_len = accum_char_len[len(accum_char_len)-1]
            elif len(accum_char_len) == cdd_mention_pos[0]+1:
                maxx_len = accum_char_len[-1] + 1000
            else:
                maxx_len = accum_char_len[cdd_mention_pos[0]+1]

            for string in modified_list:
                string_nospace = string.replace(' ', '')
                for letter in string_nospace:
                    match_result_name = pattern_name.match(letter)
                    match_result_unk = pattern_unk.match(letter)

                    if match_result_unk:
                        cdd_mention_pos_unk.append(cnt)

                    if match_result_name and maxx_len > cnt >= accum_char_len[cdd_mention_pos[0]]:
                        verify_name += 1
                        if len(cdd_mention_pos_bert_li) < 2:
                            if verify_name == 1:
                                cdd_mention_pos_bert_li.append(cnt)
                                temp_cnt = 1
                            elif verify_name == len(name):
                                cdd_mention_pos_bert_li.append(cnt)
                                verify_name = 0
                    elif verify_name == 1 and len(cdd_mention_pos_bert_li) > 0 and len(cdd_mention_pos_bert_li) != 2:
                        cdd_mention_pos_bert_li = cdd_mention_pos_bert_li[:-1]
                        verify_name = 0
                    else:
                        verify_name = 0
                cnt += 1

            if len(cdd_mention_pos_bert_li) == 0 & len(cdd_mention_pos_unk) != 0:
                cdd_mention_pos_bert_li.extend([cdd_mention_pos_unk[0], cdd_mention_pos_unk[0]+1])
            elif len(cdd_mention_pos_bert_li) != 2:
                cdd_mention_pos_bert_li = []
                cdd_mention_pos_bert_li.extend([int(cdd_mention_pos[1] * accum_char_len[-1]/sum([i for i in cdd_sent_char_lens])), int(
                    cdd_mention_pos[2] * accum_char_len[-1]/sum(i for i in cdd_sent_char_lens))])

            if cdd_mention_pos_bert_li[0] == cdd_mention_pos_bert_li[1]:
                cdd_mention_pos_bert_li[1] = cdd_mention_pos_bert_li[1]+1

            if len(cdd_sent_char_lens) == 1:
                ctx_hid.append(torch.zeros(1, CSS_hid.size(1)).to(device))
            elif cdd_mention_pos[0] == 0:
                ctx_hid.append(CSS_hid[:accum_char_len[-2]])
            else:
                ctx_hid.append(CSS_hid[accum_char_len[1]:])

            cdd_mention_pos_bert = (
                cdd_mention_pos[0], cdd_mention_pos_bert_li[0], cdd_mention_pos_bert_li[1])
            cdd_hid.append(
                CSS_hid[cdd_mention_pos_bert[1]:cdd_mention_pos_bert[2]])

        # qs hid 가 비어있을 경우, 에러 방지용
        if qs_hid == []:
            scores = '1'
            scores_false = 1
            scores_true = 1
            return scores, scores_false, scores_true

        # pooling
        qs_rep = self.pooling(qs_hid)
        ctx_rep = self.pooling(ctx_hid)
        cdd_rep = self.pooling(cdd_hid)

        # concatenate
        feature_vector = torch.cat([qs_rep, ctx_rep, cdd_rep], dim=-1)

        # dropout
        feature_vector = self.dropout(feature_vector)

        # scoring
        scores = self.mlp_scorer(feature_vector).view(-1)

        for i in unk_loc_li:
            new_element = torch.tensor([-0.9000], requires_grad=True)
            index_to_insert = i-1
            scores = torch.cat((scores[:index_to_insert], new_element, scores[index_to_insert:]), dim=0)

        scores_false = [scores[i] for i in range(scores.size(0)) if i != true_index]
        scores_true = [scores[true_index] for i in range(scores.size(0) - 1)]

        return scores, scores_false, scores_true
