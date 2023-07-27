from torch_lr_finder import LRFinder

def find_lr(model,optimizer, criterion, device,train_loader):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(
        train_loader,
        step_mode="exp",
        end_lr=10,
        num_iter=200,
    )
    mx_lr = lr_finder.plot(suggest_lr=True, skip_start=0, skip_end=0)
    lr_finder.reset()
    return mx_lr