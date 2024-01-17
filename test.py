import os
import torch
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

import config
import models


def save_log(recon_root, name_dataset, name_image, psnr, ssim, rate, consecutive=True):
    if not os.path.isfile(f"{recon_root}/Res_{name_dataset}_{rate}.txt"):
        log = open(f"{recon_root}/Res_{name_dataset}_{rate}.txt", 'w')
        log.write("=" * 120 + "\n")
        log.close()
    log = open(f"{recon_root}/Res_{name_dataset}_{rate}.txt", 'r+')
    if consecutive:
        old = log.read()
        log.seek(0)
        log.write(old)
    log.write(
        f"Res {name_image}: PSNR, {round(psnr, 2)}, SSIM, {round(ssim, 4)}\n")
    log.close()


def save_image(path, image_name, x_hat):
    recon_dataset_path = path
    recon_dataset_path_rate = f"{path}/{config.para.rate}"
    if not os.path.isdir(recon_dataset_path):
        os.mkdir(recon_dataset_path)
    if not os.path.isdir(recon_dataset_path_rate):
        os.mkdir(recon_dataset_path_rate)
    cv.imwrite(f"{recon_dataset_path_rate}/{image_name.split('.')[0]}.png", x_hat)


def testing(network, val, save_img=config.para.save, manner='grey'):
    """
    The pre-processing before TCS-Net's forward propagation and the testing platform.
    """
    recon_root = "./reconstructed_images"
    if not os.path.isdir(recon_root):
        os.mkdir(recon_root)
    datasets = [
        "BSD68"
        ]
    with torch.no_grad():
        for one_dataset in datasets:
            print(one_dataset + "reconstruction start")
            test_dataset_path = f"./dataset/{one_dataset}"
            # remove the previous log.
            # if os.path.isfile(f"{recon_root}/Res_{one_dataset}_gray_{config.para.rate}.txt"):
            #     os.remove(f"{recon_root}/Res_{one_dataset}_gray_{config.para.rate}.txt")
            # if os.path.isfile(f"{recon_root}/Res_{one_dataset}_rgb_{config.para.rate}.txt"):
            #     os.remove(f"{recon_root}/Res_{one_dataset}_rgb_{config.para.rate}.txt")

            if manner == 'grey':
                # grey manner
                sum_psnr, sum_ssim = 0., 0.
                for _, _, images in os.walk(f"{test_dataset_path}/"):
                    for one_image in images:
                        name_image = one_image.split('.')[0]
                        x = cv.imread(f"{test_dataset_path}/{one_image}", flags=cv.IMREAD_GRAYSCALE)
                        x_ori = x
                        x = torch.from_numpy(x).float()
                        x = x / 255.
                        x = (x - 0.45) / 0.22
                        h, w = x.size()

                        lack = config.para.block_size - h % config.para.block_size if h % config.para.block_size != 0 else 0
                        padding_h = torch.zeros(lack, w)
                        expand_h = h + lack
                        inputs = torch.cat((x, padding_h), 0)

                        lack = config.para.block_size - w % config.para.block_size if w % config.para.block_size != 0 else 0
                        expand_w = w + lack
                        padding_w = torch.zeros(expand_h, lack)
                        inputs = torch.cat((inputs, padding_w), 1).unsqueeze(0).unsqueeze(0)

                        inputs = torch.cat(torch.split(inputs, split_size_or_sections=config.para.block_size, dim=3), dim=0)
                        inputs = torch.cat(torch.split(inputs, split_size_or_sections=config.para.block_size, dim=2), dim=0)

                        x_hat, _ = network(inputs)

                        idx = expand_w // config.para.block_size
                        x_hat = torch.cat(torch.split(x_hat, split_size_or_sections=1 * idx, dim=0), dim=2)
                        x_hat = torch.cat(torch.split(x_hat, split_size_or_sections=1, dim=0), dim=3)
                        x_hat = x_hat.squeeze()[:h, :w]

                        x_hat = x_hat.cpu().numpy()
                        x_hat = x_hat * 0.22 + 0.45
                        x_hat = x_hat * 255.
                        x_hat = np.rint(np.clip(x_hat, 0, 255))

                        psnr = PSNR(x_ori, x_hat, data_range=255)
                        ssim = SSIM(x_ori, x_hat, data_range=255, multichannel=False)

                        sum_psnr += psnr
                        sum_ssim += ssim

                        can_save_dataset_name = one_dataset.replace("/","-")
                        if save_img:
                            save_image(f"{recon_root}/{can_save_dataset_name}_gray/", one_image, x_hat)
                        save_log(recon_root, can_save_dataset_name, name_image, psnr, ssim, f"gray_{config.para.rate}")
                    save_log(recon_root, can_save_dataset_name, None, sum_psnr / len(images), sum_ssim / len(images), f"gray_{config.para.rate}_AVG", False)
                    print(f"AVG RES of GRAY {can_save_dataset_name}: PSNR, {round(sum_psnr / len(images), 2)}, SSIM, {round(sum_ssim / len(images), 4)}")
                    if val:
                        return round(sum_psnr / len(images), 2), round(sum_ssim / len(images), 4)

            elif manner == 'rgb':
                # RGB manner
                recon_dataset_path_rgb = f"{recon_root}/{one_dataset}/"
                recon_dataset_path_rgb_rate = f"{recon_root}/{one_dataset}/{config.para.rate}"
                if not os.path.isdir(recon_dataset_path_rgb):
                    os.mkdir(recon_dataset_path_rgb)
                if not os.path.isdir(recon_dataset_path_rgb_rate):
                    os.mkdir(recon_dataset_path_rgb_rate)
                sum_psnr, sum_ssim = 0., 0.
                for _, _, images in os.walk(f"{test_dataset_path}/"):
                    for one_image in images:
                        name_image = one_image.split('.')[0]
                        x = cv.imread(f"{test_dataset_path}/{one_image}")
                        x_ori = x
                        r, g, b = cv.split(x)
                        r = torch.from_numpy(np.asarray(r)).squeeze().float() / 255.
                        g = torch.from_numpy(np.asarray(g)).squeeze().float() / 255.
                        b = torch.from_numpy(np.asarray(b)).squeeze().float() / 255.

                        x = torch.from_numpy(x).float()
                        h, w = x.size()[0], x.size()[1]

                        lack = config.para.block_size - h % config.para.block_size if h % config.para.block_size != 0 else 0
                        padding_h = torch.zeros(lack, w)
                        expand_h = h + lack
                        inputs_r = torch.cat((r, padding_h), 0)
                        inputs_g = torch.cat((g, padding_h), 0)
                        inputs_b = torch.cat((b, padding_h), 0)

                        lack = config.para.block_size - w % config.para.block_size if w % config.para.block_size != 0 else 0
                        expand_w = w + lack
                        padding_w = torch.zeros(expand_h, lack)
                        inputs_r = torch.cat((inputs_r, padding_w), 1).unsqueeze(0).unsqueeze(0)
                        inputs_g = torch.cat((inputs_g, padding_w), 1).unsqueeze(0).unsqueeze(0)
                        inputs_b = torch.cat((inputs_b, padding_w), 1).unsqueeze(0).unsqueeze(0)

                        inputs_r = torch.cat(torch.split(inputs_r, split_size_or_sections=config.para.block_size, dim=3),
                                             dim=0)
                        inputs_r = torch.cat(torch.split(inputs_r, split_size_or_sections=config.para.block_size, dim=2),
                                             dim=0)

                        inputs_g = torch.cat(torch.split(inputs_g, split_size_or_sections=config.para.block_size, dim=3),
                                             dim=0)
                        inputs_g = torch.cat(torch.split(inputs_g, split_size_or_sections=config.para.block_size, dim=2),
                                             dim=0)

                        inputs_b = torch.cat(torch.split(inputs_b, split_size_or_sections=config.para.block_size, dim=3),
                                             dim=0)
                        inputs_b = torch.cat(torch.split(inputs_b, split_size_or_sections=config.para.block_size, dim=2),
                                             dim=0)

                        r_hat, _ = network(inputs_r.to(config.para.device))
                        g_hat, _ = network(inputs_g.to(config.para.device))
                        b_hat, _ = network(inputs_b.to(config.para.device))

                        idx = expand_w // config.para.block_size
                        r_hat = torch.cat(torch.split(r_hat, split_size_or_sections=1 * idx, dim=0), dim=2)
                        r_hat = torch.cat(torch.split(r_hat, split_size_or_sections=1, dim=0), dim=3)
                        r_hat = r_hat.squeeze()[:h, :w].cpu().numpy() * 255.

                        g_hat = torch.cat(torch.split(g_hat, split_size_or_sections=1 * idx, dim=0), dim=2)
                        g_hat = torch.cat(torch.split(g_hat, split_size_or_sections=1, dim=0), dim=3)
                        g_hat = g_hat.squeeze()[:h, :w].cpu().numpy() * 255.

                        b_hat = torch.cat(torch.split(b_hat, split_size_or_sections=1 * idx, dim=0), dim=2)
                        b_hat = torch.cat(torch.split(b_hat, split_size_or_sections=1, dim=0), dim=3)
                        b_hat = b_hat.squeeze()[:h, :w].cpu().numpy() * 255.

                        r_hat, g_hat, b_hat = np.rint(np.clip(r_hat, 0, 255)), \
                                              np.rint(np.clip(g_hat, 0, 255)), \
                                              np.rint(np.clip(b_hat, 0, 255))
                        x_hat = cv.merge([r_hat, g_hat, b_hat])

                        psnr = PSNR(x_ori, x_hat, data_range=255)
                        ssim = SSIM(x_ori, x_hat, data_range=255, multichannel=True)

                        sum_psnr += psnr
                        sum_ssim += ssim

                        can_save_dataset_name = one_dataset.replace("/","-")
                        if save_img:
                            save_image(f"{recon_root}/{can_save_dataset_name}_rgb/", one_image, x_hat)
                        save_log(recon_root, can_save_dataset_name, name_image, psnr, ssim, f"rgb_{config.para.rate}")
                    save_log(recon_root, can_save_dataset_name, None, sum_psnr / len(images), sum_ssim / len(images), f"rgb_{config.para.rate}_AVG", False)
                    print(f"AVG RES of RGB {can_save_dataset_name}: PSNR, {round(sum_psnr / len(images), 2)}, SSIM, {round(sum_ssim / len(images), 4)}")
                    if val:
                        return round(sum_psnr / len(images), 2), round(sum_ssim / len(images), 4)


if __name__ == "__main__":
    my_state_dict = config.para.my_state_dict
    device = config.para.device

    net = models.ELACS_Net().eval().to(device)
    if os.path.exists(my_state_dict):
        if torch.cuda.is_available():
            trained_model = torch.load(my_state_dict, map_location=device)
        else:
            raise Exception(f"No GPU.")
        net.load_state_dict(trained_model)
    else:
        raise FileNotFoundError(f"Missing trained model of rate {config.para.rate}.")
    testing(net, val=False, save_img=True)
