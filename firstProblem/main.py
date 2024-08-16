
from utils import *


criterion = nn.BCEWithLogitsLoss()

dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=int(os.getenv('BATCH_SIZE')),
    shuffle=True)

G = Generator(int(os.getenv('Z_DIM'))).to(os.getenv('DEVICE'))
gOpt = torch.optim.Adam(G.parameters(), lr=float(os.getenv('LEARNING_RATE')))
D = Discriminator().to(os.getenv('DEVICE'))
dOpt = torch.optim.Adam(D.parameters(), lr=float(os.getenv('LEARNING_RATE')))

curStep = 0
meanGeneratorLoss = 0
meanDiscriminatorLoss = 0
testGenerator = True # Whether the generator should be tested
genLoss = False
error = False
for epoch in range(int(os.getenv('EPOCH'))):
  
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        curbatchsize = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(curbatchsize, -1).to(os.getenv('DEVICE'))

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        dOpt.zero_grad()

        # Calculate discriminator loss
        discLoss_ = discLoss(G, D, criterion, real, curbatchsize, int(os.getenv('Z_DIM')), \
                              os.getenv('DEVICE')).getLoss()

        # Update gradients
        discLoss_.backward(retain_graph=True)

        # Update optimizer
        dOpt.step()

        # For testing purposes, to keep track of the generator weights
        if testGenerator:
            oldGeneratorWeights = G.gen[0][0].weight.detach().clone()

        
        G.zero_grad()
        genLoss_ = genLoss(G, D, criterion, curbatchsize, int(os.getenv('Z_DIM')), \
                           os.getenv('DEVICE')).getLoss()
        genLoss_.backward(retain_graph=True)
        gOpt.step()
        
       

        # For testing purposes, to check that your code changes the generator weights
        if testGenerator:
            try:
                assert float(os.getenv('LEARNING_RATE')) > 0.0000002 or \
                    (G.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(G.gen[0][0].weight.detach().clone() != oldGeneratorWeights)
            except:
                error = True
                print("Runtime tests have failed")

        # Keep track of the average discriminator loss
        meanDiscriminatorLoss += discLoss_.item() / int(os.getenv('DISPLAY_STEP'))

        # Keep track of the average generator loss
        meanGeneratorLoss += genLoss_.item() / int(os.getenv('DISPLAY_STEP'))

        ### Visualization code ###
        if curStep % int(os.getenv('DISPLAY_STEP')) == 0 and curStep > 0:
            print(f"Step {curStep}: Generator loss: {meanGeneratorLoss}, discriminator loss: {meanDiscriminatorLoss}")
            fakeNoise = getNoise(curbatchsize, int(os.getenv('Z_DIM')), device=os.getenv('DEVICE'))
            fake = G(fakeNoise)
            plotter.imageShow(fake)
            plotter.imageShow(real)
            meanGeneratorLoss = 0
            meanDiscriminatorLoss = 0
        curStep += 1
