import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import utils

class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len, Q_TKN, A_TKN, SENT, EOS, MASK, TOKENIZER):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = TOKENIZER

    def __len__(self): 
        return len(self._data)

    def __getitem__(self, idx): 
        turn = self._data.iloc[idx]
        q = turn["Q"]
        a = turn["A"] 

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        if q_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)) :] 
                q_len = len(q_toked)
                a_len = self.max_len - q_len              
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len     
            if a_len <= 0:       
                q_toked = q_toked[-(int(self.max_len / 2)) :]   
                q_len = len(q_toked)
                a_len = self.max_len - q_len              
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        labels = [self.mask,] * q_len + a_toked[1:]
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return (token_ids, np.array(mask), labels_ids)

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)


def run(argv):
    Q_TKN = "<usr>"
    A_TKN = "<sys>"
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    # 허깅페이스 transformers 에 등록된 사전 학습된 koGTP2 토크나이저를 가져온다.
    koGPT2_TOKENIZER= PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, 
                                                              unk_token="<unk>", pad_token=PAD, mask_token=MASK,)
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trained_model_path = 'models/' + argv.run_model + '.pt'
    model.load_state_dict(torch.load(trained_model_path, map_location=device))

    with torch.no_grad():
        ans = ""
        old_a = ""
        while 1:
            q = input("user > ").strip()
            if q == "quit":
                break
            elif q == "":
                if len(old_a) > 30:
                    q = old_a[len(old_a)-15:]
                else:
                    q = old_a
            else:
                old_a = ""
            a = ""

            while 1:
                input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)
                pred = model(input_ids)
                pred = pred.logits
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")
                if len(a) > 30:
                    break
            a = a.replace("<unk>","")
            if old_a == "":
                print("Chatbot > {}".format(a.strip()))
                old_a = a
            else:
                print("Chatbot > {}".format(old_a + " " + a.strip()))
                old_a = old_a + " " + a



def train(argv):
    utils.preprocess(argv)
    
    Chatbot_Data = pd.read_csv('chatlogs_processed/'+argv.train_file+'_processed.csv')
    max_len=70
    Q_TKN = "<usr>"
    A_TKN = "<sys>"
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    # 허깅페이스 transformers 에 등록된 사전 학습된 koGTP2 토크나이저를 가져온다.
    koGPT2_TOKENIZER= PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, 
                                                              unk_token="<unk>", pad_token=PAD, mask_token=MASK,)
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    
    train_set = ChatbotDataset(Chatbot_Data, max_len, Q_TKN, A_TKN, SENT, EOS, MASK, koGPT2_TOKENIZER)
    train_dataloader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_batch,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    learning_rate = 3e-5
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    Sneg = -1e18

    # compiled_model = torch.compile(model)

    epoch = argv.train_epochs

    print('Start training...')
    try: 
        for epoch in range(epoch):
            print(epoch)
            for batch_idx, samples in enumerate(train_dataloader):
                optimizer.zero_grad()
                token_ids, mask, label = samples
                token_ids = token_ids.to(device)
                mask = mask.to(device)
                label = label.to(device)
                out = model(token_ids)
                out = out.logits      #Returns a new tensor with the logit of the elements of input
                mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
                mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))

                loss = criterion(mask_out.transpose(2, 1), label)
                avg_loss = loss.sum() / mask.sum()
                avg_loss.backward()
                optimizer.step()
                avg_loss_v = round(avg_loss.item(), 1)
            print('End of epoch ', str(epoch), ', Loss: ', avg_loss_v)
            torch.save(model.state_dict(), "models/" + argv.train_file + "_" + str(avg_loss_v) + ".pt")
        print('Finished training.')
        
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "models/" + argv.train_file + "_" + str(avg_loss_v) + ".pt")


        