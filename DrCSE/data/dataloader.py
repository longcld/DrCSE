from torch.utils.data import DataLoader


class Dataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(Dataloader, self).__init__(*args, **kwargs)
