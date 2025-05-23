import argparse


parser = argparse.ArgumentParser(description="Args of this repo.")
parser.add_argument("--rate", default=0.3, type=float)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--device", default="1")
parser.add_argument("--time", default=0, type=int)
parser.add_argument("--block_size", default=64, type=int)
parser.add_argument("--batch_size", default=32, type=int)

parser.add_argument("--step_num", default=16, type=int)
parser.add_argument("--num_layers", default=7, type=int)

parser.add_argument("--save", default=False)

parser.add_argument("--save_path", default=f"./trained_models")
parser.add_argument("--folder")
parser.add_argument("--my_state_dict")
parser.add_argument("--my_log")
parser.add_argument("--my_info")
para = parser.parse_args()
if para.device == "cpu":
    para.device = "cpu"
else:
    para.device = f"cuda:{para.device}"
para.folder = f"{para.save_path}/{str(int(para.rate * 100))}/"
para.my_state_dict = f"{para.folder}/num_layers_{para.num_layers}_state_dict.pth"
para.my_log = f"{para.folder}/num_layers_{para.num_layers}_log.txt"
para.my_info = f"{para.folder}/num_layers_{para.num_layers}_info.pth"
