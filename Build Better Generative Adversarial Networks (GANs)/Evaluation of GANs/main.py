from utils import *



### Loading the Pre-trained Model

transform = transforms.Compose([
    transforms.Resize(int(os.getenv('IMAGE_SIZE'))),
    transforms.CenterCrop(int(os.getenv('IMAGE_SIZE'))),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

in_coursera = False # Set this to false if you're running this outside Coursera
if in_coursera:
    import numpy as np
    data = torch.Tensor(np.load('fid_images_tensor.npz', allow_pickle=True)['arr_0'])
    dataset = torch.utils.data.TensorDataset(data, data)
else:
    dataset = CelebA(".", download=True, transform=transform)


gen = Generator(int(os.getenv('Z_DIM'))).to(os.getenv('DEVICE'))
gen.load_state_dict(torch.load(f"pretrained_celeba.pth", map_location=torch.device(os.getenv('DEVICE')))["gen"])
gen = gen.eval()

### Inception-v3 Network

inception_model = inception_v3(pretrained=False)
inception_model.load_state_dict(torch.load("inception_v3_google-1a9a5a14.pth"))
inception_model.to(os.getenv('DEVICE'))
inception_model = inception_model.eval() # Evaluation mode

### Fréchet Inception Distance

inception_model.fc = torch.nn.Identity()


### Fréchet Distance
mean = torch.Tensor([0, 0]) # Center the mean at the origin
covariance = torch.Tensor( # This matrix shows independence - there are only non-zero values on the diagonal
    [[1, 0],
     [0, 1]]
)
independent_dist = MultivariateNormal(mean, covariance)
samples = independent_dist.sample((10000,))
res = sns.jointplot(x=samples[:, 0], y=samples[:, 1], kind="kde")
plt.show()


mean = torch.Tensor([0, 0])
covariance = torch.Tensor(
    [[2, -1],
     [-1, 2]]
)
covariant_dist = MultivariateNormal(mean, covariance)
samples = covariant_dist.sample((10000,))
res = sns.jointplot(x = samples[:, 0], y =samples[:, 1], kind="kde")
plt.show()

### Putting it all together!

fake_features_list = []
real_features_list = []

gen.eval()

dataloader = DataLoader(
    dataset,
    batch_size= int(os.getenv('BATCH_SIZE')),
    shuffle=True)

cur_samples = 0
with torch.no_grad(): # You don't need to calculate gradients here, so you do this to save memory
    try:
        for real_example, _ in tqdm(dataloader, total=int(os.getenv('N_SAMPLES')) // int(os.getenv('BATCH_SIZE'))): # Go by batch
            real_samples = real_example
            real_features = inception_model(real_samples.to(os.getenv('DEVICE'))).detach().to('cpu') # Move features to CPU
            real_features_list.append(real_features)

            fake_samples = get_noise(len(real_example), int(os.getenv('Z_DIM'))).to(os.getenv('DEVICE'))
            fake_samples = preprocess(gen(fake_samples))
            fake_features = inception_model(fake_samples.to(os.getenv('DEVICE'))).detach().to('cpu')
            fake_features_list.append(fake_features)
            cur_samples += len(real_samples)
            if cur_samples > int(os.getenv('N_SAMPLES')):
                break
    except:
        print("Error in loop")
    

fake_features_all = torch.cat(fake_features_list)
real_features_all = torch.cat(real_features_list)

mu_fake = torch.mean(fake_features_all, dim=0)
mu_real = torch.mean(real_features_all, dim=0)
sigma_fake = get_covariance(fake_features_all)
sigma_real = get_covariance(real_features_all)


indices = [2, 4, 5]
fake_dist = MultivariateNormal(mu_fake[indices], sigma_fake[indices][:, indices])
fake_samples = fake_dist.sample((5000,))
real_dist = MultivariateNormal(mu_real[indices], sigma_real[indices][:, indices])
real_samples = real_dist.sample((5000,))

import pandas as pd
df_fake = pd.DataFrame(fake_samples.numpy(), columns=indices)
df_real = pd.DataFrame(real_samples.numpy(), columns=indices)
df_fake["is_real"] = "no"
df_real["is_real"] = "yes"
df = pd.concat([df_fake, df_real])
sns.pairplot(data = df, plot_kws={'alpha': 0.1}, hue='is_real')
plt.show()


with torch.no_grad():
    print(frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item())