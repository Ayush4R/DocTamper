import os
import cv2
import torch
import jpegio
import numpy as np
import pickle
import tempfile
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
import torchvision
import argparse
from dtd import *
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, required=True, help='Path to input images folder')
parser.add_argument('--mask_folder', type=str, default=None, help='Path to ground truth masks folder (optional)')
parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder')
parser.add_argument('--pth', type=str, default='dtd.pth', help='Path to model checkpoint')
parser.add_argument('--minq', type=int, default=75, help='Minimum JPEG quality')
parser.add_argument('--target_size', type=int, default=512, help='Target size for resizing images (model input size)')
parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
args = parser.parse_args()

# Create output folder if it doesn't exist
os.makedirs(args.output_folder, exist_ok=True)

# Load quantization tables
with open('qt_table.pk', 'rb') as fpk:
    pks = pickle.load(fpk)
pks_dict = {}
for k, v in pks.items():
    pks_dict[k] = torch.LongTensor(v)

# Image transformation
toctsr = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
])
totsr = ToTensorV2()


def get_jpeg_info(image_path, quality=75, target_size=512):
    """Extract DCT coefficients and quantization table from JPEG image"""
    im = Image.open(image_path)
    
    # Store original size
    w, h = im.size
    
    # Resize to fixed target size (model expects specific size)
    im = im.resize((target_size, target_size), Image.BILINEAR)
    
    # Save with specified quality to get DCT coefficients
    with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
        im_gray = im.convert("L")
        im_gray.save(tmp.name, "JPEG", quality=quality)
        jpg = jpegio.read(tmp.name)
        dct = jpg.coef_arrays[0].copy()
        
        # Get quantization table
        if quality in pks_dict:
            use_qtb = pks_dict[quality]
        else:
            # Default to quality 75 if not found
            use_qtb = pks_dict[75]
        
        im_rgb = im.convert('RGB')
    
    return im_rgb, dct, use_qtb, (w, h)


def process_image(image_path, mask_path=None, quality=75, target_size=512):
    """Process a single image and return tensors"""
    # Load and process image
    im_rgb, dct, use_qtb, orig_size = get_jpeg_info(image_path, quality, target_size)
    
    # Prepare DCT coefficients - clip and convert to int for embedding
    dct_coef = np.clip(np.abs(dct), 0, 20).astype(np.int64)
    
    # Convert to tensors
    img_tensor = toctsr(im_rgb)
    dct_tensor = torch.from_numpy(dct_coef).long()
    qtb_tensor = use_qtb
    
    # Load ground truth mask if available
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0)
        if mask is not None:
            mask = (mask != 0).astype(np.uint8)
            # Resize mask to match processed image size
            if mask.shape[0] != target_size or mask.shape[1] != target_size:
                mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            mask = totsr(image=mask)['image']
    
    return img_tensor, dct_tensor, qtb_tensor, mask, np.array(im_rgb), orig_size


def save_results(original, pred_mask, gt_mask, output_path, orig_size=None):
    """Save visualization with original, GT, and predicted mask"""
    # Resize pred_mask back to original size if needed
    if orig_size is not None:
        orig_w, orig_h = orig_size
        if pred_mask.shape[0] != orig_h or pred_mask.shape[1] != orig_w:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            # Resize original image too
            original = cv2.resize(original, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            if gt_mask is not None:
                if gt_mask.shape[0] != orig_h or gt_mask.shape[1] != orig_w:
                    gt_mask = cv2.resize(gt_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    h, w = original.shape[:2]
    
    # Prepare predicted mask visualization (white for tampered regions)
    pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pred_vis[pred_mask == 1] = [255, 255, 255]  # White for tampered
    
    if gt_mask is not None:
        # Create 3-column output: Original | GT | Prediction
        gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
        gt_vis[gt_mask == 1] = [255, 255, 255]
        
        output = np.hstack([original, gt_vis, pred_vis])
    else:
        # Create 2-column output: Original | Prediction
        output = np.hstack([original, pred_vis])
    
    cv2.imwrite(output_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


def main():
    # Get list of images
    image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(args.image_folder, ext)))
    
    if len(image_files) == 0:
        print(f"No images found in {args.image_folder}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Load model - using same approach as eval_dtd.py
    print("Loading model...")
    model = seg_dtd('', 2).cuda() if args.device == 'cuda' else seg_dtd('', 2)
    model = torch.nn.DataParallel(model)
    
    ckpt = torch.load(args.pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    # Process images
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Processing images"):
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # Check for corresponding mask
            mask_path = None
            if args.mask_folder:
                for ext in ['.png', '.jpg', '.PNG', '.JPG']:
                    potential_mask = os.path.join(args.mask_folder, base_name + ext)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        break
            
            try:
                # Process image
                img_tensor, dct_tensor, qtb_tensor, gt_mask, original_img, orig_size = process_image(
                    img_path, mask_path, args.minq, args.target_size
                )
                
                # Prepare batch (add batch dimension)
                data = Variable(img_tensor.unsqueeze(0).to(args.device))
                dct_coef = Variable(dct_tensor.unsqueeze(0).to(args.device))
                qs = Variable(qtb_tensor.unsqueeze(0).unsqueeze(1).to(args.device))
                
                # Inference
                pred = model(data, dct_coef, qs)
                pred_mask = pred.argmax(1).cpu().data.numpy()[0]
                
                # Prepare GT mask for saving
                gt_mask_np = None
                if gt_mask is not None:
                    gt_mask_np = gt_mask.numpy()
                
                # Save results
                output_path = os.path.join(args.output_folder, f"{base_name}_result.png")
                save_results(original_img, pred_mask, gt_mask_np, output_path, orig_size)
                
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                continue
    
    print(f"\nProcessing complete! Results saved to {args.output_folder}")


if __name__ == '__main__':
    main()
