import streamlit as st
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk
import re
import pickle
nltk.download('punkt_tab')
device = torch.device('cpu')
symbols = r"([+\-/*^=0123456789.,|'()])"

def tokenize(text):
    text = text.replace("\n", " NEWLINE_CHAR ")
    text = re.sub(symbols, r" \1 ", text)
    tokens = word_tokenize(text)
    tokens = ["\n" if token == "NEWLINE_CHAR" else token for token in tokens]
    tokens.append('\n')
    return tokens

def sticks(ch, output):
    if output == "":
        return ch
    else:
        if (output[-1].isnumeric() and ch.isnumeric()) or (ch in ['.', ',', '^', ')', '?', '!']) or (output[-1] in ['[', '(']):
            return ch
        else:
            return " " + ch

def generate_text(model, itos, stoi, block_size, context="", max_len=64, random_seed=None):
    with torch.no_grad():
        output = "---------\n"
        output += context
        if not context:
            words = []
            context = [0] * block_size
        else:
            words = tokenize(context)[:-1]
            context = [stoi.get(words[-block_size:][i], 0) for i in range(len(words[-block_size:]))]
            if len(context) < block_size:
                context = [0] * (block_size - len(context)) + context
        for _ in range(max_len):
            x = torch.tensor(context).view(1, -1).to(device)
            y_pred = model(x)
            if(random_seed): torch.manual_seed(random_seed)
            ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
            ch = itos[ix]
            if ch == 'END':
                break
            output += sticks(ch, output)
            context = context[1:] + [ix]
        return output

class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin_1 = nn.Linear(block_size * emb_dim, hidden_size[0])
        self.lin_2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.lin_3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.lin_4 = nn.Linear(hidden_size[2], vocab_size)
        self.activation = activation

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin_1(x))
        x = self.activation(self.lin_2(x))
        x = self.activation(self.lin_3(x))
        x = self.lin_4(x)
        return x

@st.cache_resource
def load_models():
    with open("stoi.pkl", "rb") as f:
        stoi = pickle.load(f)
    with open("itos.pkl", "rb") as f:
        itos = pickle.load(f)


    model1 = NextWord(16, len(stoi), 128, [512, 512, 512], nn.LeakyReLU(negative_slope=0.01)).to(device)    
    relu_128_16 = torch.load('LeakyReLU(negative_slope=0.01)_EMB128_Context16.pth', map_location = device)    
    relu_128_16 = {k.replace("_orig_mod.", ""): v for k, v in relu_128_16.items()}
    model1.load_state_dict(relu_128_16)
    model2 = NextWord(32, len(stoi), 128, [512, 512, 512], nn.LeakyReLU(negative_slope=0.01)).to(device)
    relu_128_32 = torch.load('LeakyReLU(negative_slope=0.01)_EMB128_Context32.pth', map_location = device)
    relu_128_32 = {k.replace("_orig_mod.", ""): v for k, v in relu_128_32.items()}
    model2.load_state_dict(relu_128_32)
    model3 = NextWord(16, len(stoi), 256, [512, 512, 512], nn.LeakyReLU(negative_slope=0.01)).to(device)
    relu_256_16 = torch.load('LeakyReLU(negative_slope=0.01)_EMB256_Context16.pth', map_location = device)
    relu_256_16 = {k.replace("_orig_mod.", ""): v for k, v in relu_256_16.items()}
    model3.load_state_dict(relu_256_16)
    model4 = NextWord(32, len(stoi), 256, [512, 512, 512], nn.LeakyReLU(negative_slope=0.01)).to(device)
    relu_256_32 = torch.load('LeakyReLU(negative_slope=0.01)_EMB256_Context32.pth', map_location = device)
    relu_256_32 = {k.replace("_orig_mod.", ""): v for k, v in relu_256_32.items()}
    model4.load_state_dict(relu_256_32)
    model5 = NextWord(16, len(stoi), 128, [512, 512, 512], nn.Tanh()).to(device)
    tanh_128_16 = torch.load('Tanh()_EMB128_Context16.pth',map_location = device)
    tanh_128_16 = {k.replace("_orig_mod.", ""): v for k, v in tanh_128_16.items()}
    model5.load_state_dict(tanh_128_16)
    model6 = NextWord(32, len(stoi), 128, [512, 512, 512], nn.Tanh()).to(device)
    tanh_128_32 = torch.load('Tanh()_EMB128_Context32.pth',map_location = device)
    tanh_128_32 = {k.replace("_orig_mod.", ""): v for k, v in tanh_128_32.items()}
    model6.load_state_dict(tanh_128_32)
    model7 = NextWord(16, len(stoi), 256, [512, 512, 512], nn.Tanh()).to(device)
    tanh_256_16 = torch.load('Tanh()_EMB256_Context16.pth',map_location = device)
    tanh_256_16 = {k.replace("_orig_mod.", ""): v for k, v in tanh_256_16.items()}
    model7.load_state_dict(tanh_256_16)
    model8 = NextWord(32, len(stoi), 256, [512, 512, 512], nn.Tanh()).to(device)
    tanh_256_32 = torch.load('Tanh()_EMB256_Context32.pth',map_location = device)
    tanh_256_32 = {k.replace("_orig_mod.", ""): v for k, v in tanh_256_32.items()}
    model8.load_state_dict(tanh_256_32)

    
    return stoi, itos, {
        'relu_128_16': model1,
        'relu_128_32': model2,
        'relu_256_16': model3,
        'relu_256_32': model4,
        'tanh_128_16': model5,
        'tanh_128_32': model6,
        'tanh_256_16': model7,
        'tanh_256_32': model8

    }

def get_model(models, activation, embedding_size, context_length):
    prefix = 'relu' if activation == "Leaky-ReLU" else 'tanh'
    key = f"{prefix}_{embedding_size}_{context_length}"
    return models[key]

def main():
    st.header("Middle School Level Questions Generator")
    
    # Load models and dictionaries
    stoi, itos, models = load_models()
    
    # UI components
    activation = st.selectbox('Activation Function', ["Leaky-ReLU", "TanH"])
    context_length = st.selectbox('Context Length', [16, 32])
    embedding_size = st.selectbox('Embedding Size', [128, 256])
    n = st.number_input(label="Enter the number of questions to be generated", min_value=1, step=1)
    max_length = st.slider("Max Length of the generated question", min_value=10, value = 64)
    random_seedd = st.number_input(label="Enter the random seed (if required, 0 indicates no random seed)", min_value=0, step=1)

    user_input = st.text_input('Enter anything you want the question to start with')


    if st.button("Generate Questions"):
      if random_seedd!=0:
          torch.manual_seed(random_seedd)
      for i in range(n):
        model_ = get_model(models, activation, embedding_size, context_length)
        generated_text = generate_text(
            model_, 
            itos, 
            stoi, 
            block_size=context_length,
            context=user_input, 
            max_len = max_length,
            random_seed = None
        )
        st.write(generated_text)

if __name__ == "__main__":
    main()