import torch
import random
import torch.nn.functional as F
import itertools
import logging


print(torch.__version__)

# read in all the words
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = { s:i+1 for i,s in enumerate(chars) }
stoi['.'] = 0
itos = { i:s for s,i in stoi.items() }

VOCAB_SIZE=27


@torch.no_grad()
def reset_model(hidden_layer=None, embedding_dimensions=None, block_size=None):
    if hidden_layer is None or embedding_dimensions is None or block_size is None:
        raise ValueError("All parameters (hidden_layer, embedding_dimensions, block_size) must be provided.")

    global g, C, W1, b1, W2, b2, parameters

    g = torch.Generator().manual_seed(2147843647)
    C = torch.randn( ( VOCAB_SIZE, embedding_dimensions ), generator=g )
    # Using the Kaiming initialization with a 5/3 gain for the tanh.
    W1 = torch.randn( ( block_size * embedding_dimensions, hidden_layer ), generator=g ) * (5/3) / (block_size * embedding_dimensions)**0.5
    b1 = torch.randn( hidden_layer, generator=g) * 0.01
    W2 = torch.randn( ( hidden_layer, VOCAB_SIZE), generator=g ) * 0.01
    b2 = torch.randn( VOCAB_SIZE, generator=g ) * 0.01
    parameters = [ C, W1, b1, W2, b2 ]

    for p in parameters:
        p.requires_grad = True


@torch.no_grad()
def build_dataset(words, block_size=None):
    if block_size is None:
        raise ValueError("All parameters (block_size) must be provided.")

    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

@torch.no_grad()
def calculate_loss( X, Y, C, block_size=None):
    if block_size is None:
        raise ValueError("All parameters (block_size) must be provided.")

    emb = C[X] # (32, 3, 2)
    h = torch.tanh( emb.view( -1, block_size * embedding_dimensions ) @ W1 + b1 ) # (32, 100)
    logits = h @ W2 + b2 # (32, 27)
    full_loss = F.cross_entropy(logits, Y)
    return full_loss.item()


@torch.no_grad()
def calculate_learning_rate( total_epochs, current_epoch, initial_lr, decay_factor=None ):
    if decay_factor is None:
        raise ValueError("All parameters (decay_factor) must be provided.")
    decay = total_epochs / decay_factor
    return initial_lr * (initial_lr ** (current_epoch / decay))


def train(X, Y, epochs=50000, batch_size=64, block_size=3, embedding_dimensions=10, initial_lr=0.1, decay_factor=2, hidden_layer=200 ):
    # Training
    lossi = []
    stepi = []

    for i in range(1, epochs + 1):
        # minibatch construct
        ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)

        # forward pass
        emb = C[X[ix]] # (batch_size, block_size, embedding_dimensions)
        h = torch.tanh( emb.view( -1, block_size * embedding_dimensions ) @ W1 + b1 ) # (batch_size, 100)
        logits = h @ W2 + b2 # (batch_size, vocab_size)
        loss = F.cross_entropy(logits, Y[ix])

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = calculate_learning_rate( epochs, i, initial_lr, decay_factor=decay_factor )
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        stepi.append(i)
        lossi.append(loss.item())

@torch.no_grad()
def calculate_losses( Xtr, Ytr, Xdev, Ydev, Xte, Yte, block_size, C ):
    training_loss = calculate_loss( Xtr, Ytr, C, block_size=block_size )
    dev_loss = calculate_loss( Xdev, Ydev, C, block_size=block_size )
    test_loss = calculate_loss( Xte, Yte, C, block_size=block_size )
    return training_loss, dev_loss, test_loss

@torch.no_grad()
def print_row( block_size, num_params, embedding_dimensions, hidden_layer, epochs, initial_lr, batch_size, decay_factor, training_loss, dev_loss, test_loss ):
    print(f'|{block_size}|{num_params}|{embedding_dimensions}|{hidden_layer}|{epochs}|{initial_lr}|{batch_size}|{decay_factor}|{training_loss:.4f}|{dev_loss:.4f}|{test_loss:.4f}|')


random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

print("| block_size | num_params | embedding_dimensions | hidden_layer | epochs | initial_lr | batch_size | decay_factor | Training Loss | Dev Loss | Test Loss |")
print("|------------|------------|----------------------|--------------|--------|------------|------------|--------------|---------------|----------|-----------|")

# epochs = 5000
# embedding_dimensions=10
# initial_lr = 0.1
# batch_size = 64
# decay_factor = 2
# hidden_layer = 200

# train(epochs=epocs,
      # block_size=block_size,
      # embedding_dimensions=embedding_dimensions,
      # initial_lr=initial_lr,
      # batch_size=batch_size,
      # decay_factor=decay_factor,
      # hidden_layer=hidden_layer
      # )

@torch.no_grad()
def get_datasets(block_size):
    cached_datasets = {}
    if block_size not in cached_datasets:
        Xtr, Ytr   = build_dataset(words[  :n1], block_size=block_size)
        Xdev, Ydev = build_dataset(words[n1:n2], block_size=block_size)
        Xte, Yte   = build_dataset(words[n2:], block_size=block_size)
        cached_datasets[block_size] = (Xtr, Ytr, Xdev, Ydev, Xte, Yte)
    return cached_datasets[block_size]


# Define the hyperparameter ranges
epoch_range = [100000]
embedding_dimensions_range = [2, 6, 10, 20]
initial_lr_values = [0.5, 0.1, 0.05]
batch_size_values = [32, 64, 128, 256]
decay_factor_values = [3, 2, 1.5, 1.25]
hidden_layer_values = [100, 200, 300, 500]
block_size = [3, 4]

hyperparameter_combinations = list(itertools.product(epoch_range, block_size, embedding_dimensions_range, initial_lr_values, batch_size_values, decay_factor_values, hidden_layer_values))
random.shuffle(hyperparameter_combinations)

# Iterate through the hyperparameter combinations
for combination in hyperparameter_combinations:
    epochs, block_size, embedding_dimensions, initial_lr, batch_size, decay_factor, hidden_layer = combination

    (Xtr, Ytr, Xdev, Ydev, Xte, Yte) = get_datasets(block_size)

    reset_model(hidden_layer=hidden_layer, embedding_dimensions=embedding_dimensions, block_size=block_size)

    train(Xtr,
          Ytr,
          epochs=epochs,
          block_size=block_size,
          embedding_dimensions=embedding_dimensions,
          initial_lr=initial_lr,
          batch_size=batch_size,
          decay_factor=decay_factor,
          hidden_layer=hidden_layer
          )
    training_loss, dev_loss, test_loss = calculate_losses( Xtr, Ytr, Xdev, Ydev, Xte, Yte, block_size, C )

    num_params = sum(p.nelement() for p in parameters)
    print_row( block_size, num_params, embedding_dimensions, hidden_layer, epochs, initial_lr, batch_size, decay_factor, training_loss, dev_loss, test_loss )
