from load_dataset import UEAloader
from utlis import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'UEA': UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    drop_last = False
    data_set = Data(
        root_path=args.root_path,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )
    return data_set, data_loader