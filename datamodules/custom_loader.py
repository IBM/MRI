
import torch
class InfiniteDataLoader_Huh:
    def __init__(self, dataset, weights, batch_size, num_workers, replacement=False, shuffle = True):
        super().__init__()

        assert weights == None
        assert not replacement

        self.loader  = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        self.init_iter()
        
    def init_iter(self):
        self._iterator = iter(self.loader)

    # def __iter__(self):
    #     while True:
    #         yield next(self._iterator)
    
    def __next__(self): #, *args, **kwargs):
        try:
            data = next(self._iterator)
        except StopIteration:
            self.init_iter()
            data = next(self._iterator)
        return data
    
