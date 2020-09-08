from pytorch_lightning import LightningDataModule

class CelebADataModule(LightningDataModule):

    name = 'CelebA'

    def __init__(
        self,
        data_dir: str,
        val_split: int = 5000,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        """
        Specs:
            - ?? classes
            - Each image is (1 x ?? x ??)
        Transforms::
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor()
            ])
        Example::
            from dataloader import CelebADataModule
            dataset = CelebADataModule('data/')
            model = StupidModel()
            Trainer().fit(model, dataset)
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
        """
        super().__init__(*args, **kwargs)
        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed

    @property
    def num_classes(self):
        """
        Return:
            ??
        """
        return -1

    def prepare_data(self):
        """
        Saves MNIST files to data_dir
        """
        MNIST(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor())
        MNIST(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size=32, transforms=None):
        """
        MNIST train set removes a subset to use for validation
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.train_transforms or self._default_transforms()

        dataset = MNIST(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, batch_size=32, transforms=None):
        """
        MNIST val set uses a subset of the training set for validation
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.val_transforms or self._default_transforms()
        dataset = MNIST(self.data_dir, train=True, download=True, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size=32, transforms=None):
        """
        MNIST test set uses the test split
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.val_transforms or self._default_transforms()

        dataset = MNIST(self.data_dir, train=False, download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def _default_transforms(self):
        if self.normalize:
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=(0.5,), std=(0.5,)),
            ])
        else:
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms