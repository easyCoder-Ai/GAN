from utils import *

criterion = nn.BCEWithLogitsLoss()


# We tranform our image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST(".", download=False, transform=transform),
    batch_size=int(os.getenv('BATCH_SIZE')),
    shuffle=True)


gen = Generator(int(os.getenv('Z_DIME'))).to(os.getenv('DEVICE'))
gen_opt = torch.optim.Adam(gen.parameters(), lr=float(os.getenv('LEARNING_RATE')), \
                           betas=(float(os.getenv('BETA_1')), float(os.getenv('BETA_2'))))
disc = Discriminator().to(os.getenv('DEVICE')) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=float(os.getenv('LEARNING_RATE')), \
                            betas=(float(os.getenv('BETA_1')), \
                                                            float(os.getenv('BETA_2'))))
# We initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
for epoch in range(int(os.getenv('EPOCHS'))):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(os.getenv('DEVICE'))

        ## Update discriminator ##
        disc_opt.zero_grad()
        fake_noise = get_noise(cur_batch_size, int(os.getenv('Z_DIME')),\
                                device=os.getenv('DEVICE'))
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / int(os.getenv('DISPLAY_STEP'))
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        ## Update generator ##
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, int(os.getenv('Z_DIME')),\
                                 device=os.getenv('DEVICE'))
        fake_2 = gen(fake_noise_2)
        disc_fake_pred = disc(fake_2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / int(os.getenv('DISPLAY_STEP'))

        ## Visualization code ##
        if cur_step % int(os.getenv('DISPLAY_STEP')) == 0 and cur_step > 0:
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1


