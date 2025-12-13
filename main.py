import argparse
import config
from data_utils import get_dataloaders
from quantum_model import AQCBottleneck
from trainer import train_model
import torch

def main(args):
    # Cập nhật config từ tham số dòng lệnh
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR = args.lr
    
    print(f"Configuration: Epochs={config.EPOCHS}, Batch={config.BATCH_SIZE}, LR={config.LR}, Frozen={args.freeze}")

    # 1. Load Data
    train_loader, val_loader = get_dataloaders(name=args.dataset, classes=[0, 1])

    # 2. Khởi tạo Model
    model = AQCBottleneck()
    
    # Logic đóng băng tham số (Frozen vs Trainable)
    mode_name = "trainable"
    if args.freeze:
        print(">>> Mode: FROZEN AQC (Parameters locked)")
        model.hamiltonian_params.requires_grad = False
        model.alpha.requires_grad = False
        mode_name = "frozen"
    else:
        print(">>> Mode: TRAINABLE AQC (Parameters learning)")

    # 3. Chạy Training
    save_file = f"{args.dataset}_{mode_name}_model.pth"
    train_model(model, train_loader, val_loader, save_name=save_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum CNN Training Script')
    
    # Các tham số có thể chỉnh từ dòng lệnh
    parser.add_argument('--dataset', type=str, default='MNIST', help='Tên bộ dữ liệu (MNIST, FashionMNIST)')
    parser.add_argument('--epochs', type=int, default=10, help='Số lượng Epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
    parser.add_argument('--freeze', action='store_true', help='Nếu có cờ này, sẽ đóng băng tham số lượng tử')
    
    args = parser.parse_args()
    main(args)