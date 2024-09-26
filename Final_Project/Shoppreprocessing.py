import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA


# Custom Dataset class for loading videos
class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None, frame_size=(224, 224), max_frames=30, apply_pca=False,
                 n_components=50):
        self.video_dir = video_dir
        self.transform = transform
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.apply_pca = apply_pca
        self.n_components = n_components
        self.video_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if
                            file.endswith(('.mp4', '.avi'))]

        # Initialize PCA if enabled
        if self.apply_pca:
            self.pca = PCA(n_components=self.n_components)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = self.load_video(video_path)

        # Apply transforms (resize, ToTensor, etc.)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)  # Stack into (num_frames, channels, height, width)

        # Apply PCA if enabled (after stacking the frames)
        if self.apply_pca:
            frames = self.apply_pca_on_frames(frames)

        return frames

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        # Create Background Subtractor for this video
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count >= self.max_frames:
                break

            # Apply background subtraction to remove static parts
            fg_mask = bg_subtractor.apply(frame)
            frame = cv2.bitwise_and(frame, frame, mask=fg_mask)  # Keep only foreground

            # Resize frame to the desired size
            frame = cv2.resize(frame, self.frame_size)

            # Convert frame to RGB and then to tensor (C, H, W)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

            frames.append(frame)
            frame_count += 1

        cap.release()

        # If less than max_frames, pad with zeros
        if len(frames) < self.max_frames:
            padding_frames = [torch.zeros_like(frames[0]) for _ in range(self.max_frames - len(frames))]
            frames.extend(padding_frames)

        return frames[:self.max_frames]

    def apply_pca_on_frames(self, frames):
        # Flatten the frames to 2D (frames, height*width*channels)
        flattened_frames = frames.view(frames.size(0), -1).numpy()  # Convert to numpy array

        # Adjust n_components to not exceed available frames
        adjusted_n_components = min(self.n_components, flattened_frames.shape[0])
        pca = PCA(n_components=adjusted_n_components)
        pca_result = pca.fit_transform(flattened_frames)

        # Convert PCA result back to torch tensor
        return torch.tensor(pca_result, dtype=torch.float32)


# Preprocessing transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    # Correct path to your video folder
    video_dir = r"C:\Users\Hp\Downloads\Shop DataSet\Shop DataSet\shop lifters"

    # Verify if the path exists
    if not os.path.exists(video_dir):
        print(f"Directory not found: {video_dir}")
    else:
        # Create dataset and dataloader
        video_dataset = VideoDataset(video_dir, transform=preprocess, apply_pca=True, n_components=50)
        dataloader = DataLoader(video_dataset, batch_size=4, shuffle=True, num_workers=4)

        # Example: Loop through data
        for batch in dataloader:
            print(batch.shape)  # Should print (batch_size, num_frames, pca_components)
