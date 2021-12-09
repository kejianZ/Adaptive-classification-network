import torch
import torchvision
import torchvision.transforms as transforms

def getLoaders(dataName, trainBatchSize = 64, testBatchSize=1000, transformation = None):
    '''

    :param dataName:Str;  dataName could be "CIFAR" or "MNIST" currently
    :return: dataloader for training data and testing data
    '''
    if transformation==None:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    dataName = dataName.upper()
    if dataName=="MNIST":
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
    elif dataName == "CIFAR":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainBatchSize,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=testBatchSize,
                                             shuffle=True)
    return trainloader, testloader
