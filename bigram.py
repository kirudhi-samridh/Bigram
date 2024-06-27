import torch
import torch.nn.functional as F

# Hyperparameters
FILE_PATH = './data/names.txt'  # Path to the input data file
NUM_EPOCHS = 1000  # Number of training epochs
LEARNING_RATE = 10  # Learning rate for gradient descent
SEED = 2147483647  # Random seed for reproducibility
PRINT_EVERY = 100  # Print loss every n epochs
GENERATION_COUNT = 20  # Number of names to generate

def prepare_data(file_path):
    """
    Prepares the data for training.
    Args:
        file_path (str): Path to the input data file.
    Returns:
        xs (Tensor): Input tensor.
        ys (Tensor): Target tensor.
        itos (dict): Index to string mapping.
        stoi (dict): String to index mapping.
    """
    words = open(file_path, 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}

    xs = []
    ys = []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            i1 = stoi[ch1]
            i2 = stoi[ch2]
            xs.append(i1)
            ys.append(i2)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return xs, ys, itos, stoi

def train_model(xs, ys, stoi):
    """
    Trains the bigram model.
    Args:
        xs (Tensor): Input tensor.
        ys (Tensor): Target tensor.
        stoi (dict): String to index mapping.
    Returns:
        W (Tensor): Trained weight matrix.
    """
    num = xs.numel()
    xenc = F.one_hot(xs, num_classes=len(stoi)).float()
    g = torch.Generator().manual_seed(SEED)
    W = torch.randn((len(stoi), len(stoi)), generator=g, requires_grad=True)

    for k in range(NUM_EPOCHS):
        # Forward pass
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(num), ys].log().mean()

        # Backward pass
        W.grad = None
        loss.backward()
        
        # Update weights
        with torch.no_grad():
            W -= LEARNING_RATE * W.grad

        # Print loss
        if k % PRINT_EVERY == 0:
            print(f'Epoch {k}, Loss: {loss.item()}')

    return W

def generate_text(W, itos, count):
    """
    Generates text using the trained model.
    Args:
        W (Tensor): Trained weight matrix.
        itos (dict): Index to string mapping.
        count (int): Number of names to generate.
    """
    g = torch.Generator().manual_seed(SEED)
    for _ in range(count):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=len(itos)).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print(''.join(out))

def main():
    """
    Main function to orchestrate the data preparation, model training, and text generation.
    """
    xs, ys, itos, stoi = prepare_data(FILE_PATH)
    W = train_model(xs, ys, stoi)
    generate_text(W, itos, GENERATION_COUNT)

if __name__ == '__main__':
    main()
