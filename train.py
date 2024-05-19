import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import MarianTokenizer
import wandb

# Set the notebook name for wandb
os.environ["WANDB_NOTEBOOK_NAME"] = "INM706-Seq2Seq_Machine_Translation.ipynb"

# Login with the API KEY
wandb.login(key="9ce954fd827fd8d839648cb3708ff788ad51bafa")

# Initialize wandb run
wandb.init(project='Translator', name='English-Albanian')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the dataset
with open('GlobalVoices.en-sq.en', 'r', encoding='utf-8') as f:
    en_sentences = f.readlines()
with open('GlobalVoices.en-sq.sq', 'r', encoding='utf-8') as f:
    sq_sentences = f.readlines()

# Verify dataset loaded correctly
print(f"English sentences sample: {en_sentences[:5]}")
print(f"Albanian sentences sample: {sq_sentences[:5]}")
print(f"Total number of sentence pairs: {len(en_sentences)}")

# Use MarianTokenizer for tokenization
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-sq')

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, tokenizer, max_length=128):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        trg = self.trg_sentences[idx]

        src_enc = self.tokenizer.encode_plus(
            src, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        trg_enc = self.tokenizer.encode_plus(
            trg, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'src': src_enc['input_ids'].squeeze(),
            'src_mask': src_enc['attention_mask'].squeeze(),
            'trg': trg_enc['input_ids'].squeeze(),
            'trg_mask': trg_enc['attention_mask'].squeeze()
        }

# Create the dataset objects
dataset = TranslationDataset(en_sentences, sq_sentences, tokenizer)

# Split the dataset into train and validation sets (90% train, 10% validation)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Data preprocessing complete.")

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        hidden = hidden.unsqueeze(0).repeat(2, 1, 1)
        cell = cell[-2:].contiguous()
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((hidden_dim * 2) + emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((hidden_dim * 2) + hidden_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        
        input = trg[0, :]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1
            
        return outputs

# Model hyperparameters
INPUT_DIM = tokenizer.vocab_size
OUTPUT_DIM = tokenizer.vocab_size
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
HID_DIM = 1024
N_LAYERS = 2
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3

wandb.config.update({
    "learning_rate": 1e-3,
    "epochs": 30,
    "batch_size": 64,
    "encoder_embedding_dim": ENC_EMB_DIM,
    "decoder_embedding_dim": DEC_EMB_DIM,
    "hidden_dim": HID_DIM,
    "num_layers": N_LAYERS,
    "encoder_dropout": ENC_DROPOUT,
    "decoder_dropout": DEC_DROPOUT
})

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
attn = Attention(HID_DIM).to(device)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn).to(device)
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
TRG_PAD_IDX = tokenizer.pad_token_id
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

def train(model, iterator, optimizer, criterion, clip, accum_steps=2):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(iterator):
        src = batch['src'].T.to(device)
        trg = batch['trg'].T.to(device)
        
        with autocast():
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg) / accum_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accum_steps
        
        preds = output.argmax(1)
        non_pad_elements = (trg != TRG_PAD_IDX).nonzero().squeeze()
        correct = preds[non_pad_elements].eq(trg[non_pad_elements]).sum().item()
        acc = correct / len(non_pad_elements)
        epoch_acc += acc
        
        wandb.log({"batch_loss": loss.item() * accum_steps, "batch_accuracy": acc})
        
        if i % 10 == 0:
            print(f'Batch {i} | Loss: {loss.item() * accum_steps:.3f} | Accuracy: {acc:.3f}')
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_trgs = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch['src'].T.to(device)
            trg = batch['trg'].T.to(device)
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            preds = output.argmax(1)
            non_pad_elements = (trg != TRG_PAD_IDX).nonzero().squeeze()
            correct = preds[non_pad_elements].eq(trg[non_pad_elements]).sum().item()
            acc = correct / len(non_pad_elements)
            epoch_acc += acc
            
            all_preds.append(preds.cpu().numpy())
            all_trgs.append(trg.cpu().numpy())
    
    all_preds = [list(map(str, sent)) for sent in all_preds]
    all_trgs = [list(map(str, sent)) for sent in all_trgs]
    bleu = bleu_score(all_preds, [[trg] for trg in all_trgs])
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator), bleu

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

N_EPOCHS = wandb.config.epochs
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss, valid_acc, bleu = evaluate(model, val_loader, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'checkpoints/seq2seq_model_epoch{epoch+1}.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Train Acc: {train_acc:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} |  Val. Acc: {valid_acc:.3f} |  Val. BLEU: {bleu:.3f}')
    
    wandb.log({"train_loss": train_loss, "train_accuracy": train_acc,
               "valid_loss": valid_loss, "valid_accuracy": valid_acc, "valid_bleu": bleu,
               "epoch": epoch + 1, "epoch_time_mins": epoch_mins, "epoch_time_secs": epoch_secs})

wandb.finish()
