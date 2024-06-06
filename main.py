import streamlit as st
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
import tempfile
from matplotlib import pyplot as plt
import cv2
from sklearn.metrics import jaccard_score
class CustomSegmentationDataset(Dataset):
    def __init__(self, im_nii_paths, transformations=None):
        self.transformations = transformations
        self.ims = self.get_slices(im_nii_paths)
        self.n_cls = 2

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im = self.ims[idx]
        if self.transformations:
            im = np.array(im)
            im = self.apply_transformations(im)
        im = self.preprocess_im(im)
        return im.float()

    def preprocess_im(self, im):
        im = torch.clamp(im, min=0)
        max_val = torch.max(im)
        if max_val > 0:
            im = im / max_val
        return im

    def get_slices(self, im_nii_paths):
        ims = []
        nii_im_data = self.read_nii(im_nii_paths)
        for idx, im in enumerate(nii_im_data):
            ims.append(im)
        return ims

    def read_nii(self, im_path):
        nii_im = nib.load(im_path)
        nii_im_data = nii_im.get_fdata().transpose(2, 1, 0)
        return nii_im_data

    def apply_transformations(self, im):
        transformed = self.transformations(image=im)
        return transformed["image"]

def load_model(model_path, n_classes=2, in_channels=1):
    model = smp.UnetPlusPlus(classes=n_classes, in_channels=in_channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, data_loader, device):
    model.eval()
    result = []
    with torch.no_grad():
        for img_batch in data_loader:
            if len(img_batch.size()) == 3:
                img_batch = img_batch.unsqueeze(1)
            img_batch = img_batch.to(device)
            output = model(img_batch)
            pred_mask = torch.softmax(output, dim=1)
            pred_mask = pred_mask.argmax(dim=1)
            result.extend(pred_mask.cpu().numpy())
    return result

def enhance_contrast(image):
    # Используем CLAHE для улучшения контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = (image * 255).astype("uint8")
    enhanced_image = clahe.apply(image)
    return enhanced_image

def visualize(img, pred_mask, true_mask=None):
    img = img.squeeze(0) if img.ndim > 2 else img
    enhanced_img = enhance_contrast(img)

    figure, ax = plt.subplots(1, 3 if true_mask is not None else 2, figsize=(18, 6))

    ax[0].imshow(enhanced_img, cmap='gist_gray')
    ax[0].title.set_text('Оригинальное изображение')

    ax[1].imshow(pred_mask, cmap='gist_gray')
    ax[1].title.set_text('Локализация опухолей')

    if true_mask is not None:
        ax[2].imshow(true_mask, cmap='gist_gray')
        ax[2].title.set_text('Эталонная маска')

    for a in ax:
        a.axis('off')
        a.set_aspect('equal')

    figure.tight_layout()
    return figure

def calculate_jaccard_index(true_mask, pred_mask, eps=1e-7):

    true_mask = np.ascontiguousarray(true_mask)
    pred_mask = np.ascontiguousarray(pred_mask)

    n_cls = np.unique(true_mask).size

    jac_per_class = []

    for c in range(n_cls):

        match_pred = pred_mask == c
        match_gt = true_mask == c

        if match_gt.sum() == 0:
            jac_per_class.append(np.nan)

        else:

            intersect = np.logical_and(match_pred, match_gt).sum()
            union = np.logical_or(match_pred, match_gt).sum()

            jac = (intersect + eps) / (union + eps)
            jac_per_class.append(jac)

    return np.nanmean(jac_per_class)

st.title('Сегментация печени на снимках КТ')

uploaded_file = st.file_uploader("Выберите изображение...", type="nii", key="file-upload")
uploaded_true_mask = st.file_uploader("Выберите эталонную маску для расчета метрики...", type="nii", key="true-mask-upload")

model_path ='UnetPlusPlus.pt'

if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "total_slices" not in st.session_state:
    st.session_state.total_slices = 0
if "true_masks" not in st.session_state:
    st.session_state.true_masks = None

def process_file(uploaded_file):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nii")
    tmp_file.write(uploaded_file.read())
    tmp_file_path = tmp_file.name
    tmp_file.close()
    return tmp_file_path

if uploaded_file is not None:
    tmp_file_path = process_file(uploaded_file)

    start_button = st.button("Запустить обработку")

    if start_button:
        transformations = A.Compose([
            A.Resize(256, 256),
            ToTensorV2(),
        ])
        dataset = CustomSegmentationDataset(tmp_file_path, transformations=transformations)

        st.session_state.dataset = dataset
        st.session_state.total_slices = len(dataset)

        model = load_model(model_path)
        device = torch.device('cpu')
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        predictions = predict(model, data_loader, device)

        st.session_state.predictions = predictions

if uploaded_true_mask is not None:
    tmp_true_file_path = process_file(uploaded_true_mask)

    transformations = A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])

    true_dataset = CustomSegmentationDataset(tmp_true_file_path, transformations=transformations)
    st.session_state.true_masks = [true_dataset[i] for i in range(len(true_dataset))]


if st.session_state.predictions is not None:
    slice_num = st.slider('Slice', 0, st.session_state.total_slices - 1, 0)

    img_slice = st.session_state.dataset[slice_num][0].unsqueeze(0)
    pred_mask = st.session_state.predictions[slice_num]

    true_mask = None
    if st.session_state.true_masks is not None:
        true_mask = st.session_state.true_masks[slice_num].numpy().squeeze()

    figure = visualize(img_slice.squeeze().cpu().numpy(), pred_mask, true_mask)
    st.pyplot(figure)

    if true_mask is not None:
        jaccard_index = calculate_jaccard_index(true_mask, pred_mask)
        st.write(f'Метрика Жакара: {jaccard_index:.4f}')
