import torch.optim as optim


def get_onecycle_scheduler(optimizer,mx_lr,train_loader,num_epochs):
    return optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=mx_lr,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=5/num_epochs,
    div_factor=100,
    three_phase=False,
    final_div_factor=100,
    anneal_strategy='linear')