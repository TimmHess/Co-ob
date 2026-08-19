import torch

"""
Function that gets a dataset and a model 
and trains the model on the dataset.
:param dataset: The dataset to train the model on.
:param model: The model to train.
:param criterion: The loss function to use during training.
:param epochs: The number of epochs to train the model for.
:param batch_size: The batch size to use during training.
"""
def train_model(
        model,
        dataset,
        optimizer,
        criterion,
        epochs,
        num_workers,
        batch_size,
        device,
        scheduler=None,
        iterations=None
    ):

    cnt_iterations = 0
    max_iterations = iterations

    # Create a dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers, 
        drop_last=True
    )

    # Train the model
    for epoch in range(epochs):
        print("Epoch:", epoch+1, "of", epochs)
        loss_mean = 0
        for _, mbatch in enumerate(train_dataloader):
            cnt_iterations += 1
            optimizer.zero_grad()

            x, y, _ = mbatch[0], mbatch[1], mbatch[-1]

            x = x.to(device)
            y = y.to(device)
            out = model(x)

            loss = criterion(out, y)
            loss_mean += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
            norm_type=2, error_if_nonfinite=True)
            
            optimizer.step()
            
            if scheduler is not None:
               scheduler.step()
            
            if max_iterations is not None: 
               if cnt_iterations >= (max_iterations -2):
                   return
            
        print("Loss:", loss_mean / (len(train_dataloader)-1))
    return