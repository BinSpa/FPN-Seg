from data import Cityscapes, URUR, GID, FBP
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'Cityscapes':
        train_set = Cityscapes.CityscapesSegmentation(args, split='train')
        val_set = Cityscapes.CityscapesSegmentation(args, split='val')
        test_set = Cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader,val_loader,test_loader,num_class
    elif args.dataset == 'URUR':
        train_set = URUR.URURSegmentation(args, split='train')
        val_set = URUR.URURSegmentation(args, split='test')
        test_set = URUR.URURSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)
        return train_loader,val_loader,test_loader,num_class
    elif args.dataset == 'GID':
        train_set = GID.GIDSegmentation(args, split='train')
        val_set = GID.GIDSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'FBP':
        train_set = FBP.FBPSegmentation(args, split='train')
        val_set = FBP.FBPSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError
