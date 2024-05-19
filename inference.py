import torch
import torch.nn as nn
from transformers import MarianTokenizer

# Load the tokenizer and model
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-sq')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model classes (Encoder, Attention, Decoder, Seq2Seq)
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

# Initialize the model
INPUT_DIM = tokenizer.vocab_size
OUTPUT_DIM = tokenizer.vocab_size
ENC_EMB_DIM = 256  # Match the dimensions used in training
DEC_EMB_DIM = 256  # Match the dimensions used in training
HID_DIM = 512      # Match the dimensions used in training
N_LAYERS = 2
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
attn = Attention(HID_DIM).to(device)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn).to(device)
model = Seq2Seq(enc, dec, device).to(device)

# Load the model checkpoint
model.load_state_dict(torch.load('/Users/enxom/Desktop/ensq/seq2seq_model_epoch8.pt', map_location=device))

def translate_sentence(sentence, tokenizer, model, device, max_len=50):
    model.eval()

    # Tokenize the sentence
    tokens = tokenizer.encode(sentence, return_tensors='pt', max_length=max_len, truncation=True, padding='max_length').to(device)
    
    # Perform inference
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(tokens.T)  

    # Prepare the input and output tensors
    trg_indexes = [tokenizer.pad_token_id]  
    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1).to(device) 

    for i in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor[-1], hidden, cell, encoder_outputs)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == tokenizer.pad_token_id:  
            break

        trg_tensor = torch.cat((trg_tensor, torch.LongTensor([pred_token]).unsqueeze(1).to(device)), dim=0)

    trg_tokens = tokenizer.decode(trg_indexes, skip_special_tokens=True)
    return trg_tokens

# Interactive translation
if __name__ == "__main__":
    while True:
        sentence = input("Enter an English sentence (or type 'quit' to exit): ")
        if sentence.lower() == 'quit':
            break
        translation = translate_sentence(sentence, tokenizer, model, device)
        print(f"Translated Sentence: {translation}")
