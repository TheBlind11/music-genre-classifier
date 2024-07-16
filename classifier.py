import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms
import torchaudio
import os
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

class MusicDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.file_list = []
        for c in self.classes:
            class_dir = os.path.join(root_dir, c)
            for track in os.listdir(class_dir):
                for sample in os.listdir(os.path.join(class_dir, track)):
                    self.file_list.append((os.path.join(class_dir, track, sample), self.classes.index(c)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        waveform, _ = self.load_and_pad_audio(file_path)
        
        spectrogram = torchaudio.transforms.MelSpectrogram(
            n_mels=64,  # Reduce the number of Mel filters
            n_fft=400,  # Adjust n_fft if needed
            hop_length=160,  # Adjust hop_length if needed
        )(waveform).squeeze(0)
        
        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label
    
    def load_and_pad_audio(self, file_path, max_duration = 5): # max_duration in seconds
        waveform, sample_rate = torchaudio.load(file_path)
        max_samples = int(max_duration * sample_rate)
        
        if waveform.size(1) < max_samples:
            pad_size = max_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.size(1) > max_samples:
            waveform = waveform[:, :max_samples]

        return waveform, sample_rate

class WaveNet(nn.Module):
    def __init__(self, num_classes, in_channels=64, channels=64, kernel_size=3, dilation_factors=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]):
        super(WaveNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_factors = dilation_factors
        
        self.conv_layers = nn.ModuleList()
        for dilation in dilation_factors:
            self.conv_layers.append(
                nn.Conv1d(in_channels, channels, kernel_size, stride=1, padding=dilation, dilation=dilation)
            )
        
        self.final_conv = nn.Conv1d(channels, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.relu(conv(x))

        x = self.final_conv(x)
        x = self.pool(x)
        x = x.squeeze(2)
        
        return x

transform = None

dataset = MusicDataset(root_dir='Segmented_genres', transform=transform)

k_folds = 3
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

num_classes = len(dataset.classes)
wavenet_config = {
    'in_channels': 64,
    'channels': 64,
    'kernel_size': 3,
    'dilation_factors': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
}

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

model = WaveNet(num_classes=num_classes, **wavenet_config).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

accuracy_values = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dataset.file_list, [label for _, label in dataset.file_list])):
    print(f'Fold [{fold_idx + 1}/{k_folds}]')
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for spectrograms, labels in tqdm(train_loader, desc="Training"):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * spectrograms.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Fold [{fold_idx + 1}/{k_folds}], Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), 'saves/wavenet_music_genre.pth')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for spectrograms, labels in tqdm(val_loader, desc="Testing"):
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    fold_accuracy = 100 * correct / total
    accuracy_values.append(fold_accuracy)
    print(f"Fold [{fold_idx + 1}/{k_folds}], Accuracy on validation set: {fold_accuracy:.2f}%")
    print("=" * 50)

avg_accuracy = sum(accuracy_values) / len(accuracy_values)
print(f'Average accuracy across {k_folds} folds: {avg_accuracy:.2f}%')

torch.save(model.state_dict(), 'wavenet_music_genre.pth')
