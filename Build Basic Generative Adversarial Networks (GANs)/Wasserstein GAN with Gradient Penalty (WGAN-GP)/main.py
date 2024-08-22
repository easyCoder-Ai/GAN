from utils import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size= int(os.getenv('BATCH_SIZE')),
    shuffle=True)


gen = Generator(int(os.getenv('Z_DIME'))).to(os.getenv('DEVICE'))
gen_opt = torch.optim.Adam(gen.parameters(), lr=float(os.getenv('LEARNING_RATE')), \
                           betas=(float(os.getenv('BETA_1')), float(os.getenv('BETA_2'))))
crit = Critic().to(os.getenv('DEVICE')) 
crit_opt = torch.optim.Adam(crit.parameters(), lr=float(os.getenv('LEARNING_RATE')), \
                            betas=(float(os.getenv('BETA_1')), \
                                                            float(os.getenv('BETA_2'))))


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(weights_init)
crit = crit.apply(weights_init)

cur_step = 0
generator_losses = []
critic_losses = []
for epoch in range(int(os.getenv('EPOCHS'))):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(os.getenv('DEVICE'))

        mean_iteration_critic_loss = 0
        for _ in range(int(os.getenv('CRIT_REPEATS'))):
            ### Update critic ###
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, int(os.getenv('Z_DIME')), \
                                   device=os.getenv('DEVICE'))
            fake = gen(fake_noise)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=os.getenv('DEVICE'), \
                                 requires_grad=True)
            gradient = get_gradient(crit, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, int(os.getenv('C_LAMBDA')))

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item() / int(os.getenv('CRIT_REPEATS'))
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        ### Update generator ###
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, int(os.getenv('Z_DIME')), \
                                   device=os.getenv('DEVICE'))
        fake_2 = gen(fake_noise_2)
        crit_fake_pred = crit(fake_2)
        
        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()

        # Update the weights
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]

        ### Visualization code ###
        if cur_step % int(os.getenv('DISPLAY_STEP')) == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-int(os.getenv('DISPLAY_STEP')):]) / int(os.getenv('DISPLAY_STEP'))
            crit_mean = sum(critic_losses[-int(os.getenv('DISPLAY_STEP')):]) / int(os.getenv('DISPLAY_STEP'))
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            show_tensor_images(fake)
            show_tensor_images(real)
            step_bins = 20
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
            )
            plt.legend()
            plt.show()

        cur_step += 1






